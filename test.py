import os
import csv
import cv2
import yaml
import torch
import models
import datasets
import argparse
import numpy as np
import torch.nn as nn
import tensorflow as tf

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from Levenshtein import distance
from train import make_dataloader
from matplotlib import pyplot as plt
import torchvision.transforms as T
import kornia as K
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prepare_testing():
    # Create a data loader for the test dataset
    test_loader = make_dataloader(config['test_dataset'], tag='test')
    
    # Load the super-resolution (SR) model
    sv_file = config['model']
    sv_file = torch.load(sv_file['load'])
    
    # Create the SR model based on the loaded model specifications
    model_sr = models.make(sv_file['model'], load_model=True).cuda()
    
    # Check the number of available GPUs
    n_gpus = torch.cuda.device_count()
    
    # If multiple GPUs are available, use DataParallel to parallelize the SR model
    # if n_gpus > 1:
    #     model_sr = nn.parallel.DataParallel(model_sr)
    
    # Load the OCR model based on the configuration
    model_ocr = models.make(config['model_ocr'])
    
    # Set the Torch RNG (Random Number Generator) state based on the loaded state
    state = sv_file['state']
    torch.set_rng_state(state)
    
    # Return the test data loader, the SR model, and the OCR model
    return test_loader, model_sr, model_ocr

def test(test_loader, model_sr, model_ocr, save_path):
    # Set the SR model to evaluation mode
    model_sr.eval()
    
    # Create a progress bar for visualizing the testing progress
    pbar = tqdm(test_loader, leave=False, desc='test')
    
    # Initialize a list to store predictions
    preds = []
    
    # Create a directory for saving the test result images
    results_path = save_path / Path('imgs')
    results_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            # If the input LR image is a list (possibly multiple LR images), move them to GPU
            if isinstance(batch['lr'], list):
                batch['lr'][0], batch['lr'][1] = batch['lr'][0].cuda(), batch['lr'][1].cuda()                
                
            # Generate super-resolved (SR) images using the SR model
            sr = model_sr(batch['lr'].cuda())         
            
            # If the SR output is a tuple, extract the relevant part (assuming it's the first element)
            if isinstance(sr, tuple):
                sr = sr[0].cuda()
                
            # Process each SR image, extract OCR predictions, and save images
            for img, lr, hr, lp, name in zip(sr, batch['lr'], batch['hr'], batch['gt'], batch['name']):
                # Convert the SR image from PyTorch tensor to NumPy array and adjust color channels
                # img = K.enhance.equalize_clahe(img, clip_limit=4.0, grid_size=(2, 2))
                img = img.cpu().numpy().transpose(1, 2, 0)          
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # lr = T.ToPILImage()(lr[3:6, :, :])
                lr = T.ToPILImage()(lr)
                hr = T.ToPILImage()(hr)
                lr.save(results_path / Path("lr_" + name))
                hr.save(results_path / Path("hr_" + name))
                lr = cv2.cvtColor((np.asarray(lr)/255.0).astype('float32'), cv2.COLOR_RGB2BGR)
                hr = cv2.cvtColor((np.asarray(hr)/255.0).astype('float32'), cv2.COLOR_RGB2BGR)
                # Use the OCR model to predict text from the SR image
                pred = model_ocr.OCR_pred(img)[0].replace('#', '')
                predLR = model_ocr.OCR_pred(lr)[0].replace('#', '')

                predHR = model_ocr.OCR_pred(hr)[0].replace('#', '')
 

                #.replace('#', '')
                # Calculate accuracy by measuring the edit distance between predicted and ground truth text
                # Append the prediction details to the 'preds' list
                preds.append({'PredSR': pred, 'PredLR': predLR, 'PredHR': predHR, 'Gt': lp, 'AccLR': len(lp) - distance(predLR, lp), 'AccHR': len(lp) - distance(predHR, lp), 'Acc': len(lp) - distance(pred, lp), 'Name': name})

                # Save the SR image
                img = Image.fromarray((cv2.cvtColor(img, cv2.COLOR_BGR2RGB)*255).astype(np.uint8))
                img.save(results_path / Path("sr_" + name))
            # break

            
        # Save the test results to a CSV file
        with open(save_path / Path('results.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = ['PredSR', 'PredLR', 'PredHR', 'Gt', 'AccLR', 'AccHR', 'Acc', 'Name'])
            writer.writeheader()
            writer.writerows(preds)

def main(config_, save_path):
    global config
    config = config_    
         
    # Call the prepare_testing function to set up testing
    test_loader, model_sr, model_ocr = prepare_testing()

    # Call the test function to perform the testing
    test(test_loader, model_sr, model_ocr, save_path)
    

if __name__ == '__main__':            
    # Create an argument parser to parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)

    # Parse the command line arguments
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Create a save_name based on the configuration file and tag
    save_name = args.save
    if save_name is not None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    
    # Create a save_path directory for saving the test results
    save_path = Path('./save') / Path(save_name) 
    save_path.mkdir(parents=True, exist_ok=True)    

    # Call the main function to start the testing process
    main(config, save_path)
