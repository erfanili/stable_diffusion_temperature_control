

from utils.load_save_utils import *

import random
import torch
import os



#enter prompt_file name
num_inference_steps = 50
prompt_file_name = 'try.txt'
prompt_dir = './prompt_files'
generation_output_dir = './generation_outputs'
num_seeds_per_prompt = 3
device = 'cuda:3'




models = ['sd1_5', 'sd1_5x', 'pixart']
pipe = load_model(model_name = 'sd1_5x', device = device)



prompts_dict = get_prompts_dict(directory = prompt_dir, file_name=prompt_file_name)
attn_dicts_save_dir = os.path.join(generation_output_dir,prompt_file_name.replace('.txt', ''),'attn_dicts')
images_save_dir = os.path.join(generation_output_dir,prompt_file_name.replace('.txt', ''),'images')

for idx, prompt in prompts_dict.items():
    
    seeds = [random.randint(0,100000) for _ in range(num_seeds_per_prompt)]
    
    for seed in seeds:
        with torch.no_grad():
            generator = torch.manual_seed(seed)
            image = pipe(prompt = prompt, num_inference_steps = num_inference_steps , generator = generator).images[0]
            attns_dict = pipe.attn_fetch_x.storage_x
            
            save_dict(data = attns_dict, directory = attn_dicts_save_dir, file_name = f'{idx}_{seed}')
            save_image(image = image, directory= images_save_dir, file_name = f'{idx}_{seed}')


# # 

# 




