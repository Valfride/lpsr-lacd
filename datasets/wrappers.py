
import re
import cv2
import torch
import numpy as np

import kornia as K
import albumentations as A

from PIL import Image
from pathlib import Path
from datasets import register
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

@register('parallel_images_lp')
class SR_paired_images_wrapper_lp(Dataset):
    def __init__(
            self,
            imgW,
            imgH,
            aug,
            image_aspect_ratio,
            background,
            lbp = False,
            EIR = False,
            test = False,
            dataset=None,
            ):
        
        self.imgW = imgW
        self.imgH = imgH
        self.background = eval(background)
        self.dataset = dataset
        self.aug = aug
        self.ar = image_aspect_ratio
        self.lbp = lbp
        self.EIR = EIR
        self.test = test
        
        self.transform = np.array([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1.0),
                A.InvertImg(always_apply=True),
                
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True, p=1.0),
                None
            ])
    
    def Open_image(self, img, cvt=True):
        img = cv2.imread(img)
        if cvt is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def padding(self, img, min_ratio, max_ratio, color = (0, 0, 0)):
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
    
    def extract_plate_numbers(self, file_path, pattern):
        # List to store extracted plate numbers
        plate_numbers = []
        
        # Open the text file
        with open(file_path, 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Search for the pattern in the current line
                matches = re.search(pattern, line)
                # If a match is found
                if matches:
                    # Extract the matched string
                    plate_number = matches.group(1)
                    # Add the extracted plate number to the list
                    plate_numbers.append(plate_number)
        
        # Return the list of extracted plate numbers
        return plate_numbers[0]
    
    def get_lbp(self, x):
        radius = 2
        n_points = 8 * radius
        METHOD = 'uniform'

        lbp = local_binary_pattern(x, n_points, radius, METHOD)
        return lbp.astype(np.uint8)
    
    def collate_fn(self, datas):
        lrs = []
        hrs = []
        gts = []
        file_name = []
        for item in datas:      
            lr = self.Open_image(item['lr'])
            hr = self.Open_image(item['hr'])
            gt = self.extract_plate_numbers(Path(item['hr']).with_suffix('.txt'), pattern=r'plate: (\w+)')
  
            if self.test:
                file_name.append(item['hr'].split('/')[-1])
            
            if self.aug is not False:
                augment = np.random.choice(self.transform, replace = True)
                if augment is not None:
                    lr = augment(image=lr)["image"]
            
            lr, _, _ = self.padding(lr, self.ar-0.15, self.ar+0.15, self.background)
            lr = resize_fn(lr, (self.imgH, self.imgW))
            hr = K.enhance.equalize_clahe(transforms.ToTensor()(Image.fromarray(hr)).unsqueeze(0), clip_limit=4.0, grid_size=(2, 2))
            hr = K.utils.tensor_to_image(hr.mul(255.0).byte())
            hr, _, _ = self.padding(hr, self.ar-0.15, self.ar+0.15, self.background)  
            hr = resize_fn(hr, (2*self.imgH, 2*self.imgW))
            
            lrs.append(lr)
            hrs.append(hr)
            gts.append(gt)
            
        lr = torch.stack(lrs, dim=0)
        hr = torch.stack(hrs, dim=0)
        
        gt = gts
        del lrs
        del hrs
        del gts
        if self.test and not self.lbp:
            return {
                'lr': lr, 'hr': hr, 'gt': gt, 'name': file_name
                    }
        else:
            return {
                'lr': lr, 'hr': hr, 'gt': gt
                }
    
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]        