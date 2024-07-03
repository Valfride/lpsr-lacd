import gc
import re
import cv2
import numpy as np
import tensorflow as tf

from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array

from pathlib import Path
from models import register

@register('ocr')
class Ocr_rodosol():
    def __init__(self, load=None):
        super().__init__()
        # Get the list of physical GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            # Set GPU memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limit GPU memory fraction for each GPU
            gpu_fraction = 2.0 # Limit to 50% of GPU memory
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_fraction * 1024))]
                )
        
        if load is not None:
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
        else:
            self.load = None

    
    def OCR_pred(self, img, fl = None, convert_to_bgr=False):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if self.padding:
            img, _, _ = padding(img, self.min_ratio, self.max_ratio, color = (0.5, 0.5, 0.5))        
        
        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
        img = img_to_array(img)        
        img = img.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        
        predictions = self.OCR(img)
        _ = gc.collect()
        plates = [''] * 1
        for task_idx, pp in enumerate(predictions):
            idx_aux = task_idx + 1
            task = self.tasks[task_idx]

            if re.match(r'^char[1-9]$', task):
                for aux, p in enumerate(pp):
                    plates[aux] += self.ocr_classes['char{}'.format(idx_aux)][np.argmax(p)]
            else:
                raise Exception('unknown task \'{}\'!'.format(task))
                
        return plates
    
    
def load_model(path):
	with open(path + '/model.json', 'r') as f:
		json = f.read()

	model = model_from_json(json)
	model.load_weights(path + '/weights.hdf5')
        
	return model

def padding(img, min_ratio, max_ratio, color = (0, 0, 0)):
	img_h, img_w = np.shape(img)[:2]

	border_w = 0
	border_h = 0
	ar = float(img_w)/img_h

	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1
			ar = float(img_w+border_w)/(img_h+border_h)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

	border_w = border_w//2
	border_h = border_h//2

	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
    
	return img, border_w, border_h
    
    
    
    
