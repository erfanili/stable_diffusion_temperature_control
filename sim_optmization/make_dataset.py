import sys
import os
# Get the parent directory path and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.load_save_utils import *
import numpy as np
import torch.nn.functional as F









maps_dir = '/home/erfan/repos/stable_diffusion_temperature_control/generation_outputs/s_t_optimization/color_train/attn_maps/sd1_5x/processor_x'
text_sa_data = load_pkl(directory='/home/erfan/repos/stable_diffusion_temperature_control/generation_outputs/s_t_optimization/color_train/text_sa/sd1_5x/processor_x',file_name='color_train')

timestep = 10
seq_length = 8
block = 'up_1'
smoothing = GaussianSmoothingX(kernel_size=3, sigma=0.5,channels=seq_length)
labels = load_json(directory='./',file_name='scores')

data = []
for key, value in labels.items():
    label = value
    # print(type(label)
    idx = int(key.split('_')[0])
    file_name = key 
    attn_maps = load_pkl(directory = maps_dir, file_name=file_name)
    subset_maps = attn_maps[block][timestep,:,1:seq_length+1]
    size = int(np.sqrt(subset_maps.shape[0]))
    subset_maps = subset_maps.transpose(0,1).reshape(seq_length,size,size)
    subset_maps = torch.from_numpy(subset_maps.astype(np.float32))
    subset_maps = F.pad(subset_maps,(1,1,1,1),mode="reflect",)
    smooth_maps = smoothing(subset_maps)
    smooth_maps = smooth_maps.numpy().reshape(seq_length,-1)
    # print(idx)
    data.append((smooth_maps,label,text_sa_data[idx]))


save_pkl(data = data, directory='./', file_name='labeled_maps_dataset')


import torch
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, data):
        # data is a list of (tensor, label) tuples
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves the tensor and label
        ca_map, label, text_sa = self.data[idx]
        return ca_map, label, text_sa

# Example usage
# data = [(tensor1, label1), (tensor2, label2), ...]
# dataset = TensorDataset(data)
from torch.utils.data import DataLoader
dataset = TensorDataset(data)
# Initialize DataLoader\
    


torch.save(dataset, 'labeled_maps_dataset.pt')

print(len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)













