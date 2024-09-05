
from scipy import spatial
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from collections import defaultdict
import gc
from datetime import datetime 
import pickle 
import os
import torch
import numpy as np 
import math
from typing import Dict
from PIL import Image


####dicts
def save_dict(dict: dict,dir: str, file_name: str):
    os.makedirs(dir, exist_ok = True)
    file_path = os.path.join(dir,file_name)
    with open (f'{file_path}', 'wb') as f:
        pickle.dump(dict, f)
        
        

def load_dict(dir:str, file_name:str):
    file_path = os.path.join(dir,file_name)
    if not os.path.exists(file_path):
        print("File does not exist.")
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    
    
###image



def format_time():
    now = datetime.now()
    dt = datetime.strptime(str(now), "%Y-%m-%d %H:%M:%S.%f")
    formatted_str = dt.strftime("%H:%M:%S")
    return formatted_str

def save_image(image,dir,file_name):

    os.makedirs(dir, exist_ok = True)
    file_path = os.path.join(dir,file_name)
    image.save(f'{file_path}_{format_time()}.jpg')
    
    
def images_to_gif(dir, output_path, duration=200, loop=1000):
    os.makedirs(dir, exist_ok=True)
    image_paths = os.listdir(dir)
    num_timesteps = len(image_paths)
    sorted_image_paths = [os.path.join(dir,f'{i}.jpg') for i in range(num_timesteps)]
    images = [Image.open(image) for image in sorted_image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],  # Add the rest of the images
        duration=duration,          # Duration for each frame (in milliseconds)
        loop=loop                   # Number of loops (0 for infinite loop)
    )

    
    
####model


def load_model(model_class, device, model_id = None):
    
    if model_id is not None:
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id)
        model.to(device)

    return model





##tensor and array


def tensors_to_cpu_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: tensors_to_cpu_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [tensors_to_cpu_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(tensors_to_cpu_numpy(item) for item in data)
    elif isinstance(data, set):
        return {tensors_to_cpu_numpy(item) for item in data}
    else:
        return data
    
    




