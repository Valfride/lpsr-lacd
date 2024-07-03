import re
import gc
import cv2
import yaml
import torch
import kornia
import losses
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from keras.models import Model
from losses import register, make
from kornia.losses import SSIMLoss
from torchvision import transforms
from torch.autograd import Variable
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-'+alphabet  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            if len(item)<1:
                continue
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def encode_char(self, char):

        return self.dict[char]
    
    def encode_list(self, text, K=7):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.
            K : the max length of texts

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # print(text)
        length = []
        all_result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:
            result = []
            if decode_flag:
                item = item.decode('utf-8','strict')
            # print(item)
            length.append(len(item))
            for i in range(K):
                # print(item)
                if i<len(item): 
                    char = item[i]
                    # print(char)
                    index = self.dict[char]
                    result.append(index)
                else:
                    result.append(0)
            all_result.append(result)
        return (torch.LongTensor(all_result))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
    
    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            t_item = t[i,:]
            char_list = []
            for i in range(t_item.shape[0]):
                if t_item[i] == 0:
                    pass
                    # char_list.append('-')
                else:
                    char_list.append(self.alphabet[t_item[i]])
                # print(char_list, self.alphabet[44])
            # print('char_list:  ' ,''.join(char_list))
            texts.append(''.join(char_list))
        # print('texts:  ', texts)
        return texts

    def decode_sa(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(text_index):
            text = ''.join([self.alphabet[i] for i in text_index[index, :]])
            texts.append(text.strip('-'))
        return texts


@register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss = nn.CrossEntropyLoss(weight=self.weight, 
                                        size_average=self.size_average, 
                                        ignore_index=self.ignore_index, 
                                        reduce=self.reduce, 
                                        reduction=self.reduction, 
                                        label_smoothing=self.label_smoothing)
        
    def forward(self, v1, v2):
        return self.loss(v1, v2)

@register('ssim_loss')
class sr_loss(nn.Module):
    def __init__(self, window_size=3, reduction='mean', padding='same'):
        super(sr_loss, self).__init__()
        self.window_size = window_size
        self.ssim = SSIMLoss(window_size)
    def forward(self, im1, im2):
        return self.ssim(im1, im2) 
    

@register('OCR_perceptual_loss')
class OCR_perceptual_loss(nn.Module):
    def __init__(self, alphabet, load=None, loss_weight=None, loss_specs=None):
        super(OCR_perceptual_loss, self).__init__()
        
        # Create loss functions using provided specifications
        self.load = load
        self.loss1 = CrossEntropyLoss()
        self.loss2 = make(loss_specs)
        self.alphabet = alphabet
        self.converter = strLabelConverter(self.alphabet)
        # Set the weight for the perceptual loss
        self.weight = loss_weight
        
        # Check if GPUs are available using TensorFlow (TF)
        if self.load:
            gpus = tf.config.experimental.list_physical_devices('GPU')
    
            if gpus:
                # Configure GPU memory growth for each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Limit GPU memory fraction for each GPU (here, 100% of GPU memory)
                gpu_fraction = 0.2 
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_fraction * 12288))]
                    )
            
            # Initialize attributes related to loading an OCR model
            self.load = Path(load)
            self.OCR = load_model(self.load.as_posix())
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(self.load.as_posix() + '/parameters.npy', allow_pickle=True).item()
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            self.padding = True
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
    
    def layout_penalty(self, pred_layout, gt_layout):
        penalty = 0
        for pred_char, gt_char in zip(pred_layout, gt_layout):
            # Check if a number is predicted instead of a letter
            if pred_char.isdigit() and gt_char.isalpha():
                penalty += 0.4
            # Check if a letter is predicted instead of a number
            elif pred_char.isalpha() and gt_char.isdigit():
                penalty += 0.5
        return penalty
    
    def OCR_pred(self, img, convert_to_bgr=True):
        preds = []
        imgs = []
        
        if self.padding and convert_to_bgr:
            for i, im in enumerate(img):
                im = np.array(im.detach().cpu().permute(1, 2, 0)*255).astype('uint8')
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                im , _, _ = padding(im, self.min_ratio, self.max_ratio, color = (127, 127, 127))
                imgs.append(im)
                
        for im in imgs:
            im = cv2.resize(im, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
            im = img_to_array(im)        
            im = im.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
            
            im = (im/255.0).astype('float')
            predictions = self.OCR(im, training=False)
    
            plates = [''] * 1
            for task_idx, pp in enumerate(predictions):
                idx_aux = task_idx + 1
                task = self.tasks[task_idx]
    
                if re.match(r'^char[1-9]$', task):
                    for aux, p in enumerate(pp):
                        plates[aux] += self.ocr_classes['char{}'.format(idx_aux)][np.argmax(p)]
                else:
                    raise Exception('unknown task \'{}\'!'.format(task))
            preds.extend(plates)   
        return preds
    
    def one_hot_encode(self, indices, num_classes):
        return F.one_hot(indices, num_classes=num_classes).float()
    
    def visually_similar_penalty(self, pred_char, gt_char, confusing_pairs):
        for pair in confusing_pairs:
            if pred_char in pair and gt_char in pair:
                return 1
        return 0
    
    def custom_cross_entropy(self, pred_one_hot, gt_one_hot, weights=None):
        if weights is not None:
            ce_loss = F.cross_entropy(pred_one_hot, gt_one_hot, weight=weights)
        else:
            ce_loss = F.cross_entropy(pred_one_hot, gt_one_hot)
        return ce_loss
    
    def forward(self, im1, im2, predsSR, gt, confusing_pairs, alpha=1.0):
        if self.load and predsSR is None:
            pred1_layout, pred2_layout = self.OCR_pred(im1, convert_to_bgr=True), gt
        else:
            pred1_layout, pred2_layout = predsSR, gt
        penalty = 0
        for pred, corr in zip(pred1_layout, pred2_layout):
            penalty += self.layout_penalty(pred, corr)
        
        penalty = penalty/len(pred1_layout)
        
        pred1, pred2 = self.converter.encode_list(pred1_layout, K=7).cuda(), self.converter.encode_list(pred2_layout, K=7).cuda()
        pred1 = self.one_hot_encode(pred1, len(self.alphabet))
        
        weights = torch.ones((pred1.shape[0], pred1.shape[2]))
        
        for i, (plate1, plate_gt) in enumerate(zip(pred1_layout, pred2_layout)):
            for char1, char_gt in zip(plate1, plate_gt):
                if self.visually_similar_penalty(char1, char_gt, confusing_pairs):
                    idx = self.converter.encode_char(char_gt)
                    weights[i][idx] += 0.5
        
        pred1 = torch.chunk(pred1, pred1.size(1), 0)
        loss1 = 0
        for (i, item) in enumerate(pred1):
            item = item.squeeze().cuda()
            gt = pred2[i,:].cuda()
            loss_item = self.custom_cross_entropy(item, gt, weights=weights[i].cuda())
            loss1 += loss_item
        loss1 = loss1/(i+1)
        loss2 = self.loss2(im1, im2)
     
        return loss1 + loss2 + alpha*penalty
    
def load_model(path):
	with open(path + '/model.json', 'r') as f:
		json = f.read()

	model = model_from_json(json)
	model.load_weights(path + '/weights.hdf5')
        
	return model

def padding(img, min_ratio, max_ratio, color = (0, 0, 0)):
    # Get the height and width of the input image
	img_h, img_w = np.shape(img)[:2]

    
    # Initialize variables for border width and height
	border_w = 0
	border_h = 0

    # Calculate the aspect ratio (width divided by height) of the input image
	ar = float(img_w)/img_h

    # Check if the aspect ratio is within the specified range [min_ratio, max_ratio]
	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

    # If the aspect ratio is less than the minimum allowed ratio (min_ratio)
	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1 # Increase the border width
			ar = float(img_w+border_w)/(img_h+border_h)

    # If the aspect ratio is greater than the maximum allowed ratio (max_ratio)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

    # Calculate half of the border width and height
	border_w = border_w//2
	border_h = border_h//2

    # Use OpenCV's copyMakeBorder function to add padding to the image
	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
    
	return img, border_w, border_h  