from utils.load_save_utils import *

import random
import torch
import os



all_models = ['sd1_5', 'sd1_5x', 'pixart', 'pixart_x']
models_with_attn_maps = ['sd1_5x', 'pixart_x']

processor_classes = {'processor_x': AttnProcessorX,
                     'processor_3': AttnProcessor3,}





prompt_files_generate = False 
if prompt_files_generate:
    
    
    device = 'cuda:0'
    model_name = 'sd1_5x'
    num_inference_steps = 20
    gen_dir = './generation_outputs'
    prompt_dir = './prompt_files/txt'
    prompt_file_name = 'texture_train'
    block_to_save = 'block_13'
    seed = 110101  
    pipe = load_model(model_name=model_name,device = device)
    processor_name = 'processor_3'
    prompt = 'a white cup and a black spoon'
    index_data = {
        'obj_1': 3,
        'obj_2': 8,
        'eos' : 9
    }
    prompt_list = get_prompt_list_by_line(directory=prompt_dir,file_name=prompt_file_name)
    prompt_list = prompt_list[:100]
    image, all_maps = generate(pipe=pipe,
                                processor_name = processor_name,
                                index_data = index_data,
                                prompt = prompt,
                                seed = seed,
                                block_to_save=block_to_save,
                                num_inference_steps=num_inference_steps)
    image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name)
    save_image(image=image,directory=image_save_dir, file_name=f'{prompt}_seed_{seed}')
    for word in ['obj_1', 'obj_2']:
        gif_save_dir = os.path.join(gen_dir,'gifs',model_name,processor_name)
        save_maps(all_maps= all_maps,
                    idx =index_data[word],
                    gif_save_dir = gif_save_dir,
                    file_name = f'{prompt}_seed_{seed}_{word}',
                    save_maps_by_time = True)







single_prompt = True
if single_prompt:
    device = 'cuda:0'
    model_name = 'sd1_5x'
    num_inference_steps = 20
    gen_dir = './generation_outputs'
    prompt_dir = './prompt_files/txt'
    prompt_file_name = 'texture_train'
    block_to_save = 'block_13'
    seed = 110101  
 
    processor_name = 'processor_3'
    
    
    prompt = 'a white cup and a black spoon'
    index_data = {
        'obj_1': 3,
        'obj_2': 8,
        'eos' : 9
    }
    image, all_maps = generate(model_name = model_name,
                               device = device,
                                processor_name = processor_name,
                                index_data = index_data,
                                prompt = prompt,
                                seed = seed,
                                block_to_save=block_to_save,
                                num_inference_steps=num_inference_steps)
    image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name)
    save_image(image=image,directory=image_save_dir, file_name=f'{prompt}_seed_{seed}')
    for word in ['obj_1', 'obj_2']:
        gif_save_dir = os.path.join(gen_dir,'gifs',model_name,processor_name)
        save_maps(all_maps= all_maps,
                    idx =index_data[word],
                    gif_save_dir = gif_save_dir,
                    file_name = f'{prompt}_seed_{seed}_{word}',
                    save_maps_by_time = True)

