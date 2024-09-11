

from utils.load_save_utils import *

import random
import torch
import os



#enter model_name from the list above


#enter prompt_file name
prompt_file_name = 'try.txt'
prompt_dir = './prompt_files'
generation_output_dir = './generation_outputs'
num_seeds_per_prompt = 3
num_inference_steps = 50
device = 'cuda:3'



all_models = ['sd1_5', 'sd1_5x', 'pixart', 'pixart_x']
models_with_attn_maps = ['sd1_5x', 'pixart_x']

model_name = 'pixart_x'
# model_name = 'sd1_5x'

pipe = load_model(model_name = model_name, device = device)




generate = True
if generate:

    prompts_list = get_prompt_list_by_line(directory = prompt_dir, file_name=prompt_file_name)
    attn_dicts_save_dir = os.path.join(generation_output_dir,model_name,prompt_file_name.replace('.txt', ''),'attn_dicts')
    images_save_dir = os.path.join(generation_output_dir,model_name,prompt_file_name.replace('.txt', ''),'images')

    for idx, prompt in enumerate(prompts_list):
        
        seeds = [random.randint(0,10000) for _ in range(num_seeds_per_prompt)]
        
        for seed in seeds:
            with torch.no_grad():
                generator = torch.manual_seed(seed)
                image = pipe(prompt = prompt, num_inference_steps = num_inference_steps , generator = generator).images[0]
                save_image(image = image, directory= images_save_dir, file_name = f'{idx}_{seed}')
                
                if model_name in models_with_attn_maps:
                    attns_dict = pipe.attn_fetch_x.maps_by_block_x()
                    save_dict(data = attns_dict, directory = attn_dicts_save_dir, file_name = f'{idx}_{seed}')


# # 

# 




