import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from train_funcs import register
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

alpha = 1.0
debug = False

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(f'Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, '
          f'Max Allocated: {max_allocated:.2f} MB, Max Reserved: {max_reserved:.2f} MB')

class MemoryProfiler:
    def __init__(self, model):
        self.model = model
        self.forward_hooks = []
        self.backward_hooks = []

    def register_hooks(self):
        for layer in self.model.children():
            forward_hook = layer.register_forward_hook(self.forward_hook_fn)
            backward_hook = layer.register_backward_hook(self.backward_hook_fn)
            self.forward_hooks.append(forward_hook)
            self.backward_hooks.append(backward_hook)

    def forward_hook_fn(self, module, input, output):
        print(f'Forward pass - Layer: {module.__class__.__name__}')
        print_gpu_memory_usage()

    def backward_hook_fn(self, module, grad_input, grad_output):
        print(f'Backward pass - Layer: {module.__class__.__name__}')
        print_gpu_memory_usage()

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()

def save_visualized_images(image1, image2, image3, output_path):
    # Get the size of the SR image
    sr_size = image2.size

    # Resize the LR and GT images to match the size of the SR image
    image1_resized = image1.resize(sr_size)
    image3_resized = image3.resize(sr_size)

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the resized LR image with "LR" as the title
    axes[0].imshow(image1_resized)
    axes[0].set_title("LR")
    axes[0].axis('off')

    # Display the SR image with "SR" as the title
    axes[1].imshow(image2)
    axes[1].set_title("SR")
    axes[1].axis('off')

    # Display the resized GT image with "GT" as the title
    axes[2].imshow(image3_resized)
    axes[2].set_title("GT")
    axes[2].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close(fig)
    
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
        print('decode_flag')
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
        
        length = []
        all_result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:
            result = []
            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for i in range(K):
                if i<len(item): 
                    char = item[i]
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

@register('PARALLEL_TRAINING')
def train_parallel(train_loader, ocr_model, sr_model, ocr_loss_fn, sr_loss_fn, ocr_opt, sr_opt, confusing_pair, *args):
    if debug:
        profiler = MemoryProfiler(sr_model)
        profiler.register_hooks()
    
    config = args[0]
    alphabet = config['alphabet']
    converter = strLabelConverter(alphabet)
    
    #set train mode for models
    ocr_model.train()
    sr_model.train()

    train_loss = []
    
    #create a progress bar for training (tqdm library)
    pbar = tqdm(train_loader, leave=False, desc='Train')
    
    # Iterate through batches in the training data
    for idx, batch in enumerate(pbar):
        # Predict on HR images ground truth
        
        text = converter.encode_list(batch['gt'], K=7)
        _, preds,_ = ocr_model(batch['hr'].cuda())    
        loss_ocr_real = 0
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = ocr_loss_fn(item, gt.cuda())
            loss_ocr_real += loss_item
        loss_ocr_real = loss_ocr_real/(i+1)
        
        # Predict on SR images fake images
        
        sr = sr_model(batch['lr'].cuda())
        _, preds,_ = ocr_model(sr.detach())
        loss_ocr_fake = 0
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = ocr_loss_fn(item, gt.cuda())
            loss_ocr_fake += loss_item
        loss_ocr_fake = loss_ocr_fake/(i+1)
        loss_ocr = loss_ocr_fake + loss_ocr_real
        
        ocr_opt.zero_grad()
        loss_ocr.backward()
        ocr_opt.step()
        
        # Predict on HR images ground truth
        
        _, preds,_ = ocr_model(batch['hr'].cuda())
        loss_ocr_real = 0
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = ocr_loss_fn(item, gt.cuda())
            loss_ocr_real += loss_item
        loss_ocr_real = loss_ocr_real/(i+1)

        # Predict on SR images Fake Images
        
        sr = sr_model(batch['lr'].cuda())
        _, preds,_ = ocr_model(sr.detach())   
        preds_all_sr = preds
        
        if not config['CM']:
            loss_ocr_fake = 0
            preds = torch.chunk(preds, preds.size(0), 0)
            for (i, item) in enumerate(preds):
                item = item.squeeze()
                gt = text[i,:]
                loss_item = F.cross_entropy(item, gt.cuda())
                loss_ocr_fake += loss_item
            loss_ocr_fake = loss_ocr_fake/(i+1)
        
        
        _, preds_all_sr = preds_all_sr.max(2)
        sim_preds_sr = converter.decode_list(preds_all_sr)
        
        # Calculate and backward loss on SR_NET
        if config['CM']:
            loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda(), sim_preds_sr, batch['gt'], confusing_pair)
        else:
            loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda()) + loss_ocr_fake
        loss_sr =  loss_sr_1
        
        sr_opt.zero_grad()
        loss_sr.backward()
        sr_opt.step()
        
        if debug:
            profiler.remove_hooks()
        
        if idx%3 == 0:
            rand_img = random.randint(0, len(batch['lr'])-1)
            image1 = transforms.ToPILImage()(batch['lr'][rand_img].to('cpu'))
            image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
            image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
            save_visualized_images(image1, image2, image3, config['MODEL_SR']['name']+config['tag_view']+'.png')
            
        train_loss.append(loss_sr.detach().item())
        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss),
                          'loss_ocr': loss_ocr.detach().item(),
                          'loss_sr': loss_sr_1.detach().item(),
                          'loss_fake': loss_ocr_fake.detach().item(),
                          'loss_real': loss_ocr_real.detach().item()})
        
    return sum(train_loss)/len(train_loss)
        
@register('PARALLEL_VALIDATION')
def validation_parallel(val_loader, ocr_model, sr_model, ocr_loss_fn, sr_loss_fn, confusing_pairs, *args):
    config = args[0]
    alphabet = config['alphabet']
    converter = strLabelConverter(alphabet)
    
    #set train mode for models
    ocr_model.eval()
    sr_model.eval()

    val_loss = []
    
    #create a progress bar for training (tqdm library)
    pbar = tqdm(val_loader, leave=False, desc='Val')
    n_correct = 0
    total = 0
    
    n_correct_sr = 0
    total_sr = 0
    
    if config['CM']:
        preds_sr_cm = []
        preds_gt_cm = []
    # Disable gradient computation during validation
    with torch.no_grad():
        
        # Iterate through batches in the training data
        
        for idx, batch in enumerate(pbar):
            text = converter.encode_list(batch['gt'], K=7).cuda()
            
            # Predict on HR images ground truth
            
            _, preds,_ = ocr_model(batch['hr'].cuda())
            preds_all=preds     
            loss_ocr_real = 0
            preds = torch.chunk(preds, preds.size(0), 0)
            for (i, item) in enumerate(preds):
                item = item.squeeze()
                gt = text[i,:]
                loss_item = ocr_loss_fn(item, gt)
                loss_ocr_real += loss_item
            loss_ocr_real = loss_ocr_real/(i+1)
            
            _, preds_all = preds_all.max(2)
            sim_preds = converter.decode_list(preds_all)
            text_label = batch['gt']
            
            for pred, target in zip(sim_preds, text_label):
                pred = pred.replace('-', '')
                if pred == target:
                    n_correct += 1
                total += 1
            
            # Predict on SR images fake images
            
            sr = sr_model(batch['lr'].cuda())
            _, preds,_ = ocr_model(sr.detach())
            loss_ocr_fake = 0
            preds = torch.chunk(preds, preds.size(0), 0)
            for (i, item) in enumerate(preds):
                item = item.squeeze()
                gt = text[i,:]
                loss_item = ocr_loss_fn(item, gt)
                loss_ocr_fake += loss_item
            loss_ocr_fake = loss_ocr_fake/(i+1)
    
            # Calculate loss on OCR
            loss_ocr = loss_ocr_fake + loss_ocr_real
            
            # Predict on SR images fake images            
            sr = sr_model(batch['lr'].cuda())
            _, preds,_ = ocr_model(sr.detach())  
            preds_all_sr = preds        
            
            if not config['CM']:
                loss_ocr_fake = 0
                preds = torch.chunk(preds, preds.size(0), 0)
                for (i, item) in enumerate(preds):
                    item = item.squeeze()
                    gt = text[i,:]
                    loss_item = F.cross_entropy(item, gt)
                    loss_ocr_fake += loss_item
                loss_ocr_fake = loss_ocr_fake/(i+1)
                
            _, preds_all_sr = preds_all_sr.max(2)
            
            #create lists for confusion matrix
            if config['CM']:
                preds_sr_cm.extend(preds_all_sr.detach().cpu().numpy())
                preds_gt_cm.extend(converter.encode_list(batch['gt']).detach().cpu().numpy())
            
            
            sim_preds_sr = converter.decode_list(preds_all_sr)
            text_label = batch['gt']
            
            for pred, target in zip(sim_preds_sr, text_label):
                pred = pred.replace('-', '')
                if pred == target:
                    n_correct_sr += 1
                total_sr += 1
            
            #NEW
            if config['CM']:
                loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda(), sim_preds_sr, batch['gt'], confusing_pairs)
            else:
                loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda()) + loss_ocr_fake
            loss_sr =  loss_sr_1
            
            if idx%3 == 0:
                rand_img = random.randint(0, len(batch['lr'])-1)
                image1 = transforms.ToPILImage()(batch['lr'][rand_img].to('cpu'))
                image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
                image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                save_visualized_images(image1, image2, image3, config['MODEL_SR']['name']+config['tag_view']+'.png')
            
            val_loss.append((loss_sr.detach().item()+loss_ocr.detach().item())/2)
            pbar.set_postfix({'loss': sum(val_loss)/len(val_loss), 'loss_fake': loss_ocr_fake.detach().item(), 'loss_real': loss_ocr_real.detach().item(), 'loss_ocr': loss_ocr.detach().item()})
            
           
        if config['CM']:
            flattened_preds = np.concatenate(preds_sr_cm)
            flattened_gts = np.concatenate(preds_gt_cm)
        
            conf_matrix = confusion_matrix(flattened_preds, flattened_gts, labels=range(len(alphabet)))
            conf_matrix_normalized = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            confusing_pairs = extract_confusing_pairs(conf_matrix_normalized, alphabet, 0.25)
            print(confusing_pairs)
        else:
            confusing_pairs = []
        
        print("\nAccuracy for HR")
        for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
            raw_pred = raw_pred.data
            pred = pred.replace('-', '')
            print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))    
        
        accuracy = (n_correct / float(total))
        print(f'accuracy: {accuracy*100:.2f}%')
        
        print("Accuracy for SR")
        for raw_pred, pred, gt in zip(preds_all_sr, sim_preds_sr, text_label):
            raw_pred = raw_pred.data
            pred = pred.replace('-', '')
            print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))    
        
        accuracy_sr = (n_correct_sr / float(total_sr))
        print(f'accuracy: {accuracy_sr*100:.2f}%')
        
        return sum(val_loss)/len(val_loss), 1-accuracy_sr, confusing_pairs

def extract_confusing_pairs(conf_matrix, class_names, threshold=10):
    confusing_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and conf_matrix[i, j] > threshold:
                confusing_pairs.append((class_names[i], class_names[j]))
    return confusing_pairs