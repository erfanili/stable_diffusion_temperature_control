from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline

from sd1_5.pipeline_stable_diffusion_x import StableDiffusionPipelineX



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
def save_dict(data: dict,directory: str, file_name: str):
    os.makedirs(directory, exist_ok = True)
    file_path = os.path.join(directory,file_name)
    with open (f'{file_path}', 'wb') as f:
        pickle.dump(data, f)
        
        

def load_dict(directory:str, file_name:str):
    file_path = os.path.join(directory,file_name)
    if not os.path.exists(file_path):
        print("File does not exist.")
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    

def get_prompts_dict(directory:str, file_name = None):
    if file_name is None:
        print('error: enter prompts file name')
    else:
        file_path = os.path.join(directory,file_name)
        with open(file_path,'r') as f:
            lines = f.readlines()
            lines = {idx:line.strip() for idx,line in enumerate(lines)}
        
        return lines
    
###image



def format_time():
    now = datetime.now()
    dt = datetime.strptime(str(now), "%Y-%m-%d %H:%M:%S.%f")
    formatted_str = dt.strftime("%H:%M:%S")
    return formatted_str

def save_image(image,directory,file_name):

    os.makedirs(directory, exist_ok = True)
    file_path = os.path.join(directory,file_name)
    image.save(f'{file_path}_{format_time()}.jpg')
    
    
def images_to_gif(directory, output_path, duration=200, loop=1000):
    os.makedirectorys(directory, exist_ok=True)
    image_paths = os.listdirectory(directory)
    num_timesteps = len(image_paths)
    sorted_image_paths = [os.path.join(directory,f'{i}.jpg') for i in range(num_timesteps)]
    images = [Image.open(image) for image in sorted_image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],  # Add the rest of the images
        duration=duration,          # Duration for each frame (in milliseconds)
        loop=loop                   # Number of loops (0 for infinite loop)
    )

    
    
####model


def load_model(model_name,device):
    if model_name== 'sd1_5':
        model_class = StableDiffusionPipeline
        scheduler_class = DDIMScheduler
        model_id = "runwayml/stable-diffusion-v1-5"
        time_step_spacing = 'linspace'
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
    elif model_name == 'sd1_5x':
        model_class = StableDiffusionPipelineX
        scheduler_class = DDIMScheduler
        model_id = "runwayml/stable-diffusion-v1-5"
        time_step_spacing = 'linspace'
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
    
    elif model_name == 'pixart':
        model_class = PixArtAlphaPipeline
        model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id)
        model.to(device)
    
    else:
        print('model not accepted')
        exit()
    
        
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
    
    




