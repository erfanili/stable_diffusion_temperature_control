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
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers import T5EncoderModel, CLIPTokenizer, T5Tokenizer
from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers import PixArtAlphaPipeline

from models.sd1_5.pipeline_stable_diffusion_x import StableDiffusionPipelineX
from models.sd1_5.pipeline_stable_diffusion_x_latent_update import  StableDiffusionPipelineX_2
from models.sd1_5.storage_sd1_5 import *
from models.pixart.pipeline_pixart_alpha_x import PixArtAlphaPipelineX
from models.pixart.storage_pixart_x import AttnFetchPixartX
from models.pixart.t5_attention_forward_x import forward_x
from models.sd1_5.clip_sdpa_attention_x import CLIPSdpaAttentionX
from jy_code.latent_update import *
# from models.pixart.latent_update import LatentUpdatetPixartX

 
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
        config = model.text_encoder.config

        for i in range(config.num_hidden_layers):
            spda_x=CLIPSdpaAttentionX(config).to(model.text_encoder.device, dtype=next(model.text_encoder.parameters()).dtype)
            new_state_dict = {}
            for key,value in spda_x.state_dict().items():
                new_state_dict[key] = model.text_encoder.text_model.encoder.layers[i].self_attn.state_dict()[key]
            spda_x.load_state_dict(new_state_dict)
            model.text_encoder.text_model.encoder.layers[i].self_attn = spda_x
    elif model_name == 'sd1_5x_conform':
        model_class = StableDiffusionPipelineX_2
        scheduler_class = DDIMScheduler
        model_id = "runwayml/stable-diffusion-v1-5"
        time_step_spacing = 'linspace'
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype = torch.float16,**kwargs)
        scheduler = scheduler_class.from_config(model.scheduler.config)
        model.scheduler = scheduler
        model.scheduler.config.timestep_spacing = time_step_spacing
        model.to(device)
        model.attn_fetch_x = AttnFetchSDX_2(positive_prompt = True)
        config = model.text_encoder.config

        for i in range(config.num_hidden_layers):
            spda_x=CLIPSdpaAttentionX(config).to(model.text_encoder.device, dtype=next(model.text_encoder.parameters()).dtype)
            new_state_dict = {}
            for key,value in spda_x.state_dict().items():
                new_state_dict[key] = model.text_encoder.text_model.encoder.layers[i].self_attn.state_dict()[key]
            spda_x.load_state_dict(new_state_dict)
            model.text_encoder.text_model.encoder.layers[i].self_attn = spda_x
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
        update_config = LatentUpdateConfig()
        model.latent_update_x = LatentUpdatePixartX(config = update_config)
        model.attn_fetch_x.positive_prompt = True
        for i in range(24):
            t5_attention = model.text_encoder.encoder.block[i].layer[0].SelfAttention
            t5_attention.forward =  forward_x.__get__(t5_attention, t5_attention.__class__)

 

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
             blocks_to_save=['block_13']):

    
    all_maps = {}
    if model_name in ['sd1_5', 'pixart']:
        generator = torch.manual_seed(seed)
        image = pipe(prompt = prompt, num_inference_steps = num_inference_steps , generator = generator).images[0]
        all_maps = {}
        return image, all_maps
    else:
        if model_name in ['sd1_5x', 'sd1_5x_conform']:
            pipe.attn_fetch_x.set_processor(unet = pipe.unet,processor_name=processor_name)
        elif model_name in ['pixart_x']:
            pipe.attn_fetch_x.set_processor(transformer = pipe.transformer,processor_name=processor_name)
        else:
            print('not a valid model')
            exit()
        generator = torch.manual_seed(seed)
        image = pipe(prompt = prompt, num_inference_steps = num_inference_steps , generator = generator).images[0]
        
    if blocks_to_save is not None:
        if model_name in ['sd1_5', 'sd1_5x']:
            try:
                if blocks_to_save[0] not in ['down_0', 'down_1', 'down_2', 
                                        'mid', 'up_1', 'up_2', 'up_3']:
                    
                    raise ValueError('block_to_save is not valid for this model.')
            except ValueError as e:
                raise

        elif model_name in ['pixart', 'pixart_x']:
            
                try:
                    if blocks_to_save[0] not in [f'block_{i}' for i in range(28)]:
                        raise ValueError('block_to_save is not valid for this model.')
                except ValueError as e:
                    raise
            
        
        for block in blocks_to_save:
            all_maps[block] =pipe.attn_fetch_x.maps_by_block()[block] 
                
    return image, all_maps   





def generate_latent_optimization(prompt,
                                pipe,
                                model_name,
                                processor_name,
                                seed,
                                num_inference_steps=50, # Number of steps to run the model
                                guidance_scale = 7.5, # Guidance scale for diffusion
                                attn_res = (16, 16), # Resolution of the attention map to apply CONFORM
                                max_iter_to_alter = 30, # Which steps to stop updating the latents
                                iterative_refinement_steps = [0, 10, 20], # Iterative refinement steps
                                refinement_steps = 20, # Number of refinement steps
                                scale_factor = 20, # Scale factor for the optimization step
                                do_smoothing = True, # Apply smoothing to the attention maps
                                smoothing_sigma = 0.5, # Sigma for the smoothing kernel
                                smoothing_kernel_size = 3, # Kernel size for the smoothing kernel
                                temperature = 0.5, # Temperature for the contrastive loss
                                softmax_normalize = False, # Normalize the attention maps
                                softmax_normalize_attention_maps = False, # Normalize the attention maps
                                add_previous_attention_maps = True, # Add previous attention maps to the loss calculation
                                previous_attention_map_anchor_step = None, # Use a specific step as the previous attention map
                                loss_fn = "ntxent", # Loss function to use
                                index_data=None,
                                indices_list=None,
                                use_conform=False,
                                update_latent=True):
    
    if model_name in ['sd1_5', 'sd1_5x', 'sd1_5x_conform']:
        pipe.attn_fetch_x.set_processor(unet = pipe.unet,processor_name=processor_name,index_data = index_data)
    elif model_name in ['pixart', 'pixart_x']:
        pipe.attn_fetch_x.set_processor(transformer = pipe.transformer,processor_name=processor_name,index_data = index_data)
    else:
        print('not a valid model')
        exit()
    
    steps_to_save_attention_maps = list(range(num_inference_steps))
    
    
    if len(indices_list.keys()) == 4:
        token_groups = [
            [indices_list['attr1'], indices_list['obj1']],
            [indices_list['attr2'], indices_list['obj2']],  
            # [4],
            # [5]
            # [4,5]
        ]
    elif len(indices_list.keys()) == 2:
        token_groups = [
            [indices_list['obj1']],
            [indices_list['obj2']]
        ]
    elif len(indices_list.keys()) == 3:
        token_groups = [
            [indices_list['obj1']],
            [indices_list['attr2'],indices_list['obj2']]
        ]
    # token_groups = [
    #     [1,2],
    #     [1,2]
    # ]
    
    if not update_latent:
        max_iter_to_alter = 0
        iterative_refinement_steps = []
    
    # images, attention_maps, cos_dict = pipe(    
    images, attention_maps = pipe(
        prompt=prompt,
        token_groups=token_groups,
        indices_list=indices_list,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed),
        num_inference_steps=num_inference_steps,
        max_iter_to_alter=max_iter_to_alter,
        attn_res=attn_res,
        scale_factor=scale_factor,
        iterative_refinement_steps=iterative_refinement_steps,
        steps_to_save_attention_maps=steps_to_save_attention_maps,
        do_smoothing=do_smoothing,
        smoothing_sigma=smoothing_sigma,
        smoothing_kernel_size=smoothing_kernel_size,
        temperature=temperature,
        refinement_steps=refinement_steps,
        softmax_normalize=softmax_normalize,
        softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        add_previous_attention_maps=add_previous_attention_maps,
        previous_attention_map_anchor_step=previous_attention_map_anchor_step,
        loss_fn=loss_fn,
        conform=use_conform
    )
    return images, attention_maps
######prompt

def save_text_sa_avg(text_sa,
                 directory,
                 file_name,
                 eos_idx=None):

    os.makedirs(directory,exist_ok=True)
    # print(text_sa)

    text_sa = text_sa.detach().cpu().numpy()
    text_sa = text_sa[1:eos_idx,1:eos_idx]*255*2
    # text_sa = reshape_n_scale_array(text_sa,size)
    im = Image.fromarray(text_sa.astype(np.uint16))
    im = im.resize((256,256))
    if im.mode == 'I;16':
        im = im.convert('L')
    save_image(image=im,directory=directory, file_name = file_name)


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
    save_pkl(data = all_maps, directory =gif_save_dir, file_name = file_name)
    if save_maps_by_time:
        for t in range(num_inference_steps):
            directory = os.path.join(gif_save_dir,'maps',f'{idx}')
            save_image(image=idx_maps[t],directory=directory, file_name = f'{t}')

def save_text_sa(text_sa,
                 directory,
                 file_name,
                 block_to_save = 'block_11'):

    os.makedirs(directory,exist_ok=True)


    text_sa = text_sa[block_to_save].detach().cpu().numpy()
    text_sa = text_sa[1:20,1:20]*255*2
    # text_sa = reshape_n_scale_array(text_sa,size)
    im = Image.fromarray(text_sa.astype(np.uint16))
    # im = im.resize((256,256))
    if im.mode == 'I;16':
        im = im.convert('L')
    save_image(image=im,directory=directory, file_name = file_name)



def reshape_n_scale_array(array,size):
    maximum = (np.median(array)+1e-3)*10
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

    return words, indices




def attend_n_excite_word_n_index_animals(prompt,tokenizer):
    words = {}
    token_ids ={}
    indices = {}
    words = prompt.split(' ')
    index_of_and = words.index('and')
    noun1_idx = 1
    noun2_idx = index_of_and+2


    noun1 = words[noun1_idx].strip('\n. ')
    noun2 = words[noun2_idx].strip('\n. ')  
    words = {'noun1': noun1,'noun2': noun2}
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

    return words, indices


def attend_n_excite_word_n_index_animals_objects(prompt,tokenizer):
    words = {}
    token_ids ={}
    indices = {}
    words = prompt.split(' ')
    if 'with' not in words:
        index_of_and = words.index('and')
        noun1_idx = 1
        adj2_idx = index_of_and+2
        noun2_idx = adj2_idx+1
        noun1 = words[noun1_idx].strip('\n. ')
        adj2 = words[adj2_idx]
        noun2 = words[noun2_idx].strip('\n. ')  
        words = {'noun1': noun1, 'adj2': adj2, 'noun2': noun2}
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
        return words, indices
    else:
        words = {}
        token_ids ={}
        indices = {}
        words = prompt.split(' ')
        index_of_and = words.index('with')
        noun1_idx = 1
        noun2_idx = index_of_and+2


        noun1 = words[noun1_idx].strip('\n. ')
        noun2 = words[noun2_idx].strip('\n. ')  
        words = {'noun1': noun1,'noun2': noun2}
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
    if file_name.split('.')[-1] == 'pkl':
        file_path = os.path.join(directory,file_name)
    else:
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


def get_text_attn_map(prompt,tokenizer, text_encoder):
    # model_name = "t5-small"  # You can change to other variants like t5-base or t5-large
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # text_encoder = T5EncoderModel.from_pretrained(model_name)
    # os.makedirs(directory, exist_ok=True)
    # prompt = prompt
    
    
    token_ids = get_token_ids(prompt=prompt,tokenizer = tokenizer)
    # decoded_ids = decode_token_ids(token_ids,tokenizer=tokenizer)

    token_ids = token_ids.to(text_encoder.device)


    embeddings = text_encoder(token_ids)
    attn_maps = text_encoder.encoder.attn_maps_x
    
    
    return attn_maps



import matplotlib.pyplot as plt
import imageio
import numpy as np





def map_visualization(all_maps,
                      idx,
                      save_dir,
                      file_name,
                      num_inference_steps = 20):


    # Save each tensor visualization as an image
    for t in range(num_inference_steps):
        map_t = all_maps[t,:,idx]
        size = int(np.sqrt(len(map_t)))
        map_t = map_t.reshape((size,size)).astype(np.uint16)
        # Plot the tensor as a heatmap
        plt.imshow(map_t, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title(f'Index {idx}, Time Step {t}')
        # Save the image
        map_save_dir = os.path.join(save_dir,'visualization','maps',file_name)
        os.makedirs(map_save_dir,exist_ok=True)
        plt.savefig(f'{map_save_dir}/{t}.png')
        plt.close()  # Close the plot to avoid displaying it in the notebook
    # Create a GIF from the images

        with imageio.get_writer(f'{file_name}.gif', mode='I', duration=0.1) as writer:
            for t in range(num_inference_steps):
                path = os.path.join(save_dir,'visualization','gifs',f'{file_name}',f'{t}')
                os.makedirs(path,exist_ok=True)
                image = imageio.imread(f'{path}.png')
                writer.append_data(image)
            
            
# def save_maps(all_maps,
#               idx,
#               gif_save_dir = './gifs',
#               file_name = f'{0}',
#               save_maps_by_time = False,
#               num_inference_steps=20):
#     idx_maps = maps_to_images(attn_array = all_maps,idx = idx)
#     os.makedirs(gif_save_dir,exist_ok=True)
#     save_gif(images = idx_maps, directory=gif_save_dir, file_name = file_name)
#     if save_maps_by_time:
#         for t in range(num_inference_steps):
#             directory = os.path.join(gif_save_dir,'maps',f'{idx}')
#             save_image(image=idx_maps[t],directory=directory, file_name = f'{t}')




class GaussianSmoothingX(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # print(input.size())
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)



