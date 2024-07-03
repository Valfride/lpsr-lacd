from datasets import register
from torch.utils.data import Dataset
from pathlib import Path

@register('parallel_training')
class parallel_training(Dataset):
    def __init__(self, path_split, phase='training'):
        self.split_file = path_split
        self.phase = phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()    
        for path in data:    
            path_hr, path_lr, split = path.split(';')
            
            path_hr = path_hr.strip()
            path_lr = path_lr.strip()
            
            sample = {"lr": path_lr,
                      "hr": path_hr
                      }
            
            if self.phase in split:
                self.dataset.append(sample)
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]