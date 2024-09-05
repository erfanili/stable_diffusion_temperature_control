import torch
import os
from diffusers import StableDiffusionPipeline
from sd1_5.pipeline_stable_diffusion_x import StableDiffusionPipelineX
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from modules.unet_2d_condition_x import UNet2DConditionModel
import pickle
from sd1_5.attention_processor_x import AttnProcessorX
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from utils.load_save_utils import *
import random



model_id = "runwayml/stable-diffusion-v1-5"
model_class = StableDiffusionPipelineX
device = 'cuda:3'
time_step_spacing = 'linspace'
scheduler_class = DDIMScheduler
generation_output_dir = './generation_outputs'
prompt_dir = './prompt_files'
#enter prompt_file name
prompt_file_name = 'try.txt'
num_seeds_per_prompt = 3
num_inference_steps = 50


pipe = load_model(model_class=model_class, model_id = model_id, device = device)
scheduler = scheduler_class.from_config(pipe.scheduler.config)
pipe.scheduler.config.timestep_spacing = time_step_spacing


def get_prompts_dict(dir:str, file_name = None):
    if file_name is None:
        print('error: enter prompts file name')
    else:
        file_path = os.path.join(dir,file_name)
        with open(file_path,'r') as f:
            lines = f.readlines()
            lines = {idx:line.strip() for idx,line in enumerate(lines)}
        
        return lines

prompts_dict = get_prompts_dict(dir = prompt_dir, file_name=prompt_file_name)
attn_dicts_save_dir = os.path.join(generation_output_dir,prompt_file_name.replace('.txt', ''),'attn_dicts')
images_save_dir = os.path.join(generation_output_dir,prompt_file_name.replace('.txt', ''),'images')

for idx, prompt in prompts_dict.items():
    
    seeds = [random.randint(0,100000) for _ in range(num_seeds_per_prompt)]
    
    for seed in seeds:
        with torch.no_grad():
            generator = torch.manual_seed(seed)
            image = pipe(prompt = prompt, num_inference_steps = num_inference_steps , generator = generator).images[0]
            attns_dict = pipe.attn_fetch_x.storage_x
            
            save_dict(dict = attns_dict, dir = attn_dicts_save_dir, file_name = f'{idx}_{seed}')
            save_image(image = image, dir = images_save_dir, file_name = f'{idx}_{seed}')


# # 

# 




