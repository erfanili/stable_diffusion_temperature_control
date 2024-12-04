from utils.load_save_utils import *

import random
import torch
import os
from models.processors import AttnProcessor3, AttnProcessorX, AttnProcessor4
from models.pixart.t5_attention_forward_x import forward_x

all_models = ['sd1_5', 'sd1_5x', 'pixart', 'pixart_x','sd1_5x_conform']
models_with_attn_maps = ['sd1_5x', 'pixart_x','sd1_5x_conform']

processor_classes = {'processor_x': AttnProcessorX,
                     'processor_3': AttnProcessor3,
                     'processor_4': AttnProcessor4,}


single_prompt = True
if single_prompt:
    device = 'cuda:0'
    num_inference_steps = 20
    gen_dir = './generation_outputs'
    prompt_dir = './prompt_files/txt'
    prompt_file_name = 'texture_train'
    block_to_save = 'up_1'
    seed = 110101  
    
    processor_name = 'processor_3'
    
    model_name = 'sd1_5x'
    image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name)
    gif_save_dir = os.path.join(gen_dir,'gifs',model_name,processor_name)
    pipe = load_model(model_name=model_name,device = device)

    # prompt = 'Beneath the sprawling canopy of ancient trees, whose branches twisted and turned like the pages of a forgotten story, a soft breeze whispered secrets carried from distant lands, rustling the golden leaves that carpeted the forest floor, while a curious squirrel darted between the shadows, pausing only briefly to inspect an acorn with the air of a seasoned treasure hunter, as the faint sound of a bubbling brook echoed in the background, mingling with the melodious chirping of birds, creating a symphony of nature that seemed to blur the boundaries between reality and dreams.'
    # prompt = 'As the sun dipped below the horizon, painting the sky in hues of orange, pink, and violet, a solitary train rumbled along the tracks, its rhythmic clatter filling the evening air, while passengers inside leaned against fogged-up windows, lost in thought or buried in books, unaware that the small raindrops beginning to patter against the glass were the first signs of a sudden autumn storm that would soon sweep through the quiet countryside, drenching fields, stirring the leaves into frantic swirls, and leaving behind the distinct, earthy scent of rain on soil.'
    prompt = 'a plastic bag and a metallic desk lamp'
    index_data = {
        'obj_1': 3,
        'obj_2': 8,
        'eos' : 9
    }


    image, all_maps = generate(prompt = prompt,
                                pipe = pipe,
                                model_name = model_name,
                                processor_name = processor_name,
                                index_data = index_data,
                                seed = seed,
                                block_to_save=block_to_save,
                                num_inference_steps=num_inference_steps)
    all_text_maps = pipe.attn_fetch_x.store_text_sa(text_encoder = pipe.text_encoder)

    # print(attn_data['block_0'].size())
    

    save_image(image=image,directory=image_save_dir, file_name=f'{prompt}_seed_{seed}')
    for word in ['obj_1', 'obj_2']:
        gif_save_dir = os.path.join(gen_dir,'gifs',model_name,processor_name)
        save_maps(all_maps= all_maps,
                    idx =index_data[word],
                    gif_save_dir = gif_save_dir,
                    file_name = f'{prompt}_seed_{seed}_{word}',
                    save_maps_by_time = True)
        save_dir = os.path.join(gen_dir,model_name)
        # map_visualization(all_maps= all_maps,
        #             idx =index_data[word],
        #             save_dir = save_dir,
        #             file_name = f'{prompt}_seed_{seed}_{word}',
        #             num_inference_steps=num_inference_steps)

    text_sa_save_dir = os.path.join(gen_dir,'text_sa',model_name)
    save_text_sa(text_sa = all_text_maps,
                 directory = text_sa_save_dir,
                 file_name = f'{prompt}',
                 block_to_save = 'block_0')



prompt_files_generate = False
if prompt_files_generate:
    
    
    device = 'cuda:0'
    model_name = 'sd1_5x'
    num_inference_steps = 20
    gen_dir = './generation_outputs'
    prompt_dir = './prompt_files/txt'
    prompt_file_name = 'texture_train'
    block_to_save = 'up_3'
    seed = 110101  
   
    processor_name = 'processor_3'
    prompt = 'a white cup and a black spoon'
    index_data = {
        'obj_1': 3,
        'obj_2': 8,
        'eos' : 9
    }
    prompt_list = get_prompt_list_by_line(directory=prompt_dir,file_name=prompt_file_name)
    prompt_list = prompt_list[:100]
    pipe = load_model(model_name=model_name,device = device)
 
    for prompt in prompt_list:
        image, all_maps = generate(prompt = prompt,
                                   pipe = pipe,
                                    model_name = model_name,
                                    processor_name = processor_name,
                                    index_data = index_data,
                                    seed = seed,
                                    block_to_save=block_to_save,
                                    num_inference_steps=num_inference_steps)
        image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name)
        save_image(image=image,directory=image_save_dir, file_name=f'{prompt}_seed_{seed}')
        for word in ['obj_1', 'obj_2']:
            
            save_maps(all_maps= all_maps,
                        idx =index_data[word],
                        gif_save_dir = gif_save_dir,
                        file_name = f'{prompt}_seed_{seed}_{word}',
                        save_maps_by_time = True)



analyze_maps = True
if analyze_maps:
    device = 'cuda:0'
    num_inference_steps = 20
    gen_dir = './generation_outputs'
    prompt_dir = './prompt_files/txt'
    prompt_file_name = 'texture_train'
    block_to_save = 'up_1'
    seed = 110101  
    
    processor_name = 'processor_3'
    
    model_name = 'sd1_5x'
    prompt = 'a plastic bag and a metallic desk lamp'
    
    gif_save_dir = os.path.join(gen_dir,'gifs',model_name,processor_name)
    all_maps = load_pkl(directory = gif_save_dir, file_name = f'{prompt}_seed_{seed}_obj_1')
    for idx in range(30):
        array = all_maps[19,:,idx]
        plt.hist(array,bins = 25)
        plt.savefig(f'{gif_save_dir}/{idx}')
        plt.close()



