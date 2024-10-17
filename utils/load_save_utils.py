import torch 
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime 
import pickle 
import os
import torch
import numpy as np 
from PIL import Image
import json
from typing import Dict

from transformers import T5EncoderModel, CLIPTokenizer, T5Tokenizer
from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers import PixArtAlphaPipeline

from models.sd1_5.pipeline_stable_diffusion_x import StableDiffusionPipelineX
from models.sd1_5.storage_sd1_5 import AttnFetchSDX
from models.pixart.pipeline_pixart_alpha_x import PixArtAlphaPipelineX
from models.pixart.storage_pixart_x import AttnFetchPixartX
from models.processors import AttnProcessor3, AttnProcessorX



 
def load_model(model_name,device, **kwargs):

    if model_name== 'sd1_5':
        model_class = StableDiffusionPipeline
        scheduler_class = DDIMScheduler
        model_id = "runwayml/stable-diffusion-v1-5"
        time_step_spacing = 'linspace'
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype = torch.float16,**kwargs)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
    elif model_name == 'sd1_5x':
        model_class = StableDiffusionPipelineX
        scheduler_class = DDIMScheduler
        model_id = "runwayml/stable-diffusion-v1-5"
        time_step_spacing = 'linspace'
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype = torch.float16,**kwargs)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
        model.attn_fetch_x = AttnFetchSDX(positive_prompt = True)

    elif model_name == 'pixart':
        model_class = PixArtAlphaPipeline
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype = torch.float16,**kwargs)
        model.to(device)

    elif model_name == 'pixart_x':
        model_class = PixArtAlphaPipelineX
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id,torch_dtype = torch.float16)
        model.to(device)
        model.attn_fetch_x = AttnFetchPixartX()
        model.attn_fetch_x.positive_prompt = True

    else:
        print('model not accepted')
        exit()
    
        
    return model

def generate(prompt,
             pipe,
             model_name,
             processor_name,
             index_data,
             seed,
             num_inference_steps,
             block_to_save='block_13'):
    
    if model_name in ['sd1_5', 'sd1_5x']:
        try:
            if block_to_save not in ['down_0', 'down_1', 'down_2', 
                                    'mid', 'up_1', 'up_2', 'up_3']:
                
                raise ValueError('block_to_save is not valid for this model.')
        except ValueError as e:
            raise

    elif model_name in ['pixart', 'pixart_x']:
        try:
            if block_to_save not in [f'block_{i}' for i in range(28)]:
                raise ValueError('block_to_save is not valid for this model.')
        except ValueError as e:
            raise
        
    
    

    processor_classes = {'processor_x':AttnProcessorX,
                'processor_3': AttnProcessor3,}
    processor = processor_classes[processor_name](idx1 =index_data['obj_1'], idx2 = index_data['obj_2'],eos_idx = index_data['eos'])

    if model_name in ['sd1_5', 'sd1_5x']:
        pipe.attn_fetch_x.set_processor(unet = pipe.unet,processor=processor)
    elif model_name in ['pixart', 'pixart_x']:
        pipe.attn_fetch_x.set_processor(transformer = pipe.transformer,processor=processor)
    else:
        print('not a valid model')
        exit()
    generator = torch.manual_seed(seed)
    image = pipe(prompt = prompt, num_inference_steps = num_inference_steps , generator = generator).images[0]
    all_maps = pipe.attn_fetch_x.maps_by_block()[block_to_save]    

    return image, all_maps

######prompt


def format_time():
    now = datetime.now()
    dt = datetime.strptime(str(now), "%Y-%m-%d %H:%M:%S.%f")
    formatted_str = dt.strftime("%H:%M:%S")
    return formatted_str

def save_image(image,directory,file_name):

    os.makedirs(directory, exist_ok = True)
    file_path = os.path.join(directory,file_name)
    image.save(f'{file_path}_{format_time()}.jpg')
    return image
    
def save_gif(images,directory,file_name, duration=200, loop=1000):

    save_path = os.path.join(directory,file_name)+'.gif'
    mode = images[0].mode
    size = images[0].size
    white_frame = Image.new(mode, size, color="white")

    # Prepend the white frame to the image list
    images.insert(0, white_frame)
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],  # Add the rest of the images
        duration=duration,          # Duration for each frame (in milliseconds)
        loop=loop                   # Number of loops (0 for infinite loop)
    )
    return images[0]


def save_maps(all_maps,
              idx,
              gif_save_dir = './gifs',
              file_name = f'{0}',
              save_maps_by_time = False,
              num_inference_steps=20):
    idx_maps = maps_to_images(attn_array = all_maps,idx = idx)
    os.makedirs(gif_save_dir,exist_ok=True)
    save_gif(images = idx_maps, directory=gif_save_dir, file_name = file_name)
    if save_maps_by_time:
        for t in range(num_inference_steps):
            directory = os.path.join(gif_save_dir,'maps',f'{idx}')
            save_image(image=idx_maps[t],directory=directory, file_name = f'{t}')



def reshape_n_scale_array(array,size):
    maximum = (np.median(array)+1e-3)*5
    array = (array - np.min(array))/maximum*255
    array = array.reshape((size,size)).astype(np.uint16)
    return array



    
def maps_to_images(attn_array,idx):
    map_list = []
    for t in range(len(attn_array)):
        
        map_t = attn_array[t,:,idx]
        size = int(np.sqrt(len(map_t)))
        map_t = reshape_n_scale_array(map_t,size)
        im = Image.fromarray(map_t)
        im = im.resize((256,256))
        if im.mode == 'I;16':
            im = im.convert('L')
        map_list.append(im)
    return map_list
    



def get_token_ids(prompt,tokenizer,max_length = 120):
    """
    outputs a tensor of 1*len_prompt of token_ids.
    """
    token_ids = tokenizer(text = prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",)['input_ids']
    
    idx = get_eos_idx(token_ids)
    token_ids = token_ids[:,:idx+1] # tensor shape 1, eos_idx
    
    return token_ids


def get_eos_idx(token_ids):
    """
    outputs the eos index, which is equal to the seq length of the prompt.
    """
    idx = torch.nonzero(token_ids[0]==1).item()
    
    return idx

def decode_token_ids(token_ids:torch.Tensor,tokenizer):
    """
    decodes the token_ids tensor of shape 1*seq_len into a list of strings.
    """
    labels = []
    for id in token_ids[0]:
        labels.append(tokenizer.decode(id))

    return labels

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
    prompt_tokens = tokenizer.simply_tokenize(text = prompt)
    prompt_tokens = prompt_tokens.tolist()[0]
    pointer = 0
    for key ,word in words.items():
        token_ids = tokenizer.tokenize_a_word(word)

        if (isinstance(tokenizer.tokenizer, T5Tokenizer) and word in ('spherical', 'oblong')):
            first_id = token_ids[1]
            num_tokens = len(token_ids)-1
        else:
            first_id = token_ids[0]
            num_tokens = len(token_ids)

        first_index = prompt_tokens[pointer:].index(first_id)
        indices[key] = list(range(pointer+first_index,pointer+first_index+num_tokens))
        pointer += first_index
        print(indices)
    return words, indices



class MyTokenizer():
    def __init__(self,model_name,device = 'cuda:0'):
        self.pipe = load_model(model_name=model_name,device = device, transformer = None)
        self.tokenizer = self.pipe.tokenizer
        self.device = device
        self.encoder = self.pipe.text_encoder
        if model_name == 'pixart':
            self.max_length = 120
        else:
            self.max_length = 77
        
    def tokenize_a_word(self,word):
        
        if isinstance(self.tokenizer,CLIPTokenizer):
            tokens = self.tokenizer(text = word,add_special_tokens = True)['input_ids'][1:-1]
        elif isinstance(self.tokenizer,T5Tokenizer):
            tokens = self.tokenizer(text = word,add_special_tokens = True)['input_ids'][:-1]
            
        return tokens

    def simply_tokenize(self,text):
        
        tokens = self.tokenizer(text = text,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",)['input_ids']
        result = tokens
  
        return result
    
    def decode_a_token_id(self,token_list:list):
        
        word = self.tokenizer.decode(token_list)
        
        return word
    


####dicts
def save_pkl(data: dict,directory: str, file_name: str):
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
    os.makedirs(directory, exist_ok=True)
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
    file_names = sorted([os.path.splitext(file)[0] for file in files])
    
    return file_names
###image



def to_cpu_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: to_cpu_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_cpu_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu_numpy(item) for item in data)
    else:
        return data
    
    
def rearrange_by_layers(data:dict) -> Dict:
    layers = list(list(data.values())[0].keys())
    rearranged_output = {}
    for layer in layers:
        attn_tensors = torch.stack([data[time][layer] for time in data.keys()])
        rearranged_output[layer] = attn_tensors


    return rearranged_output


def get_text_attn_map(prompt,directory = './'):
    model_name = "t5-small"  # You can change to other variants like t5-base or t5-large
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    text_encoder = T5EncoderModel.from_pretrained(model_name)
    os.makedirs(directory, exist_ok=True)
    prompt = prompt
    
    
    token_ids = get_token_ids(prompt=prompt,tokenizer = tokenizer)
    decoded_ids = decode_token_ids(token_ids,tokenizer=tokenizer)

    


    embeddings = text_encoder(token_ids)
    attn_maps = text_encoder.encoder.attn_maps_x
    
    
    return attn_maps