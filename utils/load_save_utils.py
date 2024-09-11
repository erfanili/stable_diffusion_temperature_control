from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import T5EncoderModel, CLIPTokenizer, T5Tokenizer

from diffusers import PixArtAlphaPipeline

from models.sd1_5.pipeline_stable_diffusion_x import StableDiffusionPipelineX
from models.sd1_5.storage_sd1_5 import AttnFetchSDX

from models.pixart.pipeline_pixart_alpha_x import PixArtAlphaPipelineX
from models.pixart.storage_pixart import AttnFetchPixartX

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
import json

######prompt


def get_prompt_words_n_indices(prompt,tokenizer):
    words = {}
    token_ids ={}
    indices = {}
    words = prompt.split(' ')
    index_of_and = words.index('and')
    if words[0] in ('a' , 'an'):
        adj1_idx = 1
    else:
        adj1_idx = 0
    if words[index_of_and+1] in ('a' , 'an'):
        adj2_idx = index_of_and+2
    else:
        adj2_idx = index_of_and+1
    noun1_idx = list(range(adj1_idx+1,index_of_and))
    noun2_idx = list(range(adj2_idx+1,len(words)))
    adj1 = words[adj1_idx]
    noun1 = (' ').join([words[_] for _ in noun1_idx]).strip('\n. ')
    adj2 = words[adj2_idx]
    noun2 = (' ').join(words[_] for _ in noun2_idx).strip('\n. ')  
    words = {'adj1': adj1, 'noun1': noun1, 'adj2': adj2, 'noun2': noun2}
    # token_ids = {'adj1': tokenizer.tokenize_a_word(adj1),
    #              'noun1': tokenizer.tokenize_a_word(noun1),
    #              'adj2': tokenizer.tokenize_a_word(adj2),
    #              'noun2': tokenizer.tokenize_a_word(noun2)}

    prompt_tokens = tokenizer.simply_tokenize(text = prompt)

    pointer = 0
    for key in words.keys():

        token_ids = tokenizer.tokenize_a_word(words[key])
        first_id = token_ids[0]
        num_tokens = len(token_ids)
        first_index = prompt_tokens[pointer:].index(first_id)
        indices[key] = list(range(pointer+first_index,pointer+first_index+num_tokens))
        pointer += first_index
    return words, indices

class MyTokenizer():
    def __init__(self,model_name = 'sd1_5',device = 'cuda:0'):
        self.pipe = load_model(model_name=model_name, device = device)
        self.tokenizer = self.pipe.tokenizer
        
        
    def tokenize_a_word(self,word):
        if isinstance(self.tokenizer,CLIPTokenizer):
            tokens = self.tokenizer(text = word)['input_ids'][1:-1]
        elif isinstance(self.tokenizer,T5Tokenizer):
            # print(tokenizer.__class__.__name__)
            tokens = self.tokenizer(text = word)['input_ids'][:-1]
            
        return tokens
            
    def simply_tokenize(self,text):
        tokens = self.tokenizer(text = text)['input_ids']
    
    
        return tokens
    
    def decode_a_token_id(self,token_list:list):
        if isinstance(self.tokenizer,CLIPTokenizer):
            word = self.tokenizer.decode(token_list)
        # elif isinstance(self.tokenizer,T5Tokenizer):
        #     # print(tokenizer.__class__.__name__)
        #     tokens = self.tokenizer(text = word)['input_ids'][:-1] 
        
        return word
    
def tokenize_one_word(word):
    pipe = load_model(model_name='sd1_5',device = 'cuda:0')
    tokenizer = pipe.tokenizer
    # print(tokenizer.__class__.__name__)

####dicts
def save_dict(data: dict,directory: str, file_name: str):
    os.makedirs(directory, exist_ok = True)
    file_path = os.path.join(directory,file_name)
    with open (f'{file_path}.pkl', 'wb') as f:
        pickle.dump(data, f)
        
        

def load_pkl(directory:str, file_name:str):
    file_path = os.path.join(directory,file_name)+'.pkl'
    if not os.path.exists(file_path):
        print("File does not exist.")
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    

def get_prompt_list_by_line(directory:str, file_name = None):
    if file_name is None:
        print('error: enter prompts file name')
    else:
        file_path = os.path.join(directory,file_name)+'.txt'
        with open(file_path,'r') as f:
            lines = f.read().splitlines()
     
        return lines
    

def save_json(directory:str, file_name:str, data:dict):
    output_path = os.path.join(directory,file_name)+'.json'
    with open(output_path, 'w+') as fp:
        json.dump(obj=data,fp=fp, indent = 2, separators=(',', ': '))


def load_json(directory:str, file_name:str):
    file_path = os.path.join(directory,file_name)+'.json'
    if not os.path.exists(file_path):
        print("File does not exist.")
    else:
        with open(file_path, 'rb') as f:
            data = json.load(f)
        return data


def get_file_names(directory):

    files = os.listdir(directory)
    file_names = [os.path.splitext(file)[0] for file in files]
    
    return file_names
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
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype = torch.float16)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
    elif model_name == 'sd1_5x':
        model_class = StableDiffusionPipelineX
        scheduler_class = DDIMScheduler
        model_id = "runwayml/stable-diffusion-v1-5"
        time_step_spacing = 'linspace'
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype = torch.float16)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
        model.attn_fetch_x = AttnFetchSDX(positive_prompt = True)
        model.attn_fetch_x.set_processor_x(model.unet)
    
    elif model_name == 'pixart':
        model_class = PixArtAlphaPipeline
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype = torch.float16)
        model.to(device)

    elif model_name == 'pixart_x':
        model_class = PixArtAlphaPipelineX
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype = torch.float16)
        model.to(device)
        model.attn_fetch_x = AttnFetchPixartX(positive_prompt = True)
        model.attn_fetch_x.set_processor_x(model.transformer)
    
    else:
        print('model not accepted')
        exit()
    
        
    return model





###attn

def resahpe_n_scale_array(array,size):
    array = (array - np.min(array))/(np.max(array) -np.min(array))*255
    array = array.reshape((size,size)).astype(np.uint16)
    return array


def save_attn_by_layer(attn_array, token, output_dir, output_name):
    attn = attn_array[:,token]
    size = int(math.sqrt(len(attn)))
    attn = resahpe_n_scale_array(attn,size)
    print(attn.shape)

    os.makedirs(output_dir, exist_ok = True)
    file_path = os.path.join(output_dir,output_name)
    im = Image.fromarray(attn)
    im = im.resize((256,256))
    if im.mode == 'I;16':
        im = im.convert('L')
    im.save(file_path)
    