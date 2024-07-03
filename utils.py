import os
import time
import shutil
import math
import random

import numpy as np

from math import exp
from pathlib import Path
import torch
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def make_optimizer(model_param, spec, load_optimizer=False):
    Optimizer = {
        'adam': Adam
        }[spec['name']]
    optimizer = Optimizer(model_param, **spec['args'])
    
    if load_optimizer:
        optimizer.load_state_dict(spec['sd'])
        
    return optimizer

_log_path = None
def set_log_path(path):
    global _log_path
    _log_path = Path(path)
    
def log(obj, filename=Path('log.txt')):
    # print(obj)
    if _log_path is not None:
        with open(_log_path / filename, 'a') as f:
            print(obj, file=f)
            
def make_log_writer(save_path):
    set_log_path(save_path)
    writer = SummaryWriter(save_path / Path('tensorboard'))
    
    return log, writer
    
    
class Timer():
    def __init__(self):
        self.timer = time.time()
         
    def _set(self):
        self.timer = time.time()
        
    def _get(self):
        return time.time()-self.timer
    
    def time_text(self, time):
        if time >= 3600:
            return '{:.1f}h'.format(time/3600)
        elif time >=60:
            return '{:.1f}m'.format(time/60)
        else:
            return '{:.1f}s'.format(time)
  
def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

class Early_stopping():
    def __init__(self, patience=5, min_delta=0, counter=0, min_validation_loss=np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = counter
        self.min_validation_loss = min_validation_loss
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            
            return False, True
            
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        
        return False, False

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0
        
    def add(self, v, n=1.0):
        self.v = (self.v*self.n+v*n)/(self.n+n)
        self.n += n
        
    def item(self):
        return self.v
