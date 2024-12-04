from utils.load_save_utils import *

import random
import torch
import os
from models.processors import AttnProcessor3, AttnProcessorX, AttnProcessorX_2
import torch.nn.functional as F

all_models = ['sd1_5', 'sd1_5x', 'pixart', 'pixart_x']
models_with_attn_maps = ['sd1_5x', 'pixart_x']

processor_classes = {'processor_x': AttnProcessorX,
                     'processor_3': AttnProcessor3,
                     'processor_x_2': AttnProcessorX_2,
                     }


prompt_files_generate_test = True
if prompt_files_generate_test:
    
    # save dir
    prompt_file_name = 'animals_objects'
    gen_dir = f'./generation_outputs_our_test/{prompt_file_name}'
    cos_sim_dir = './cross_attn'
    prompt_dir = './prompt_files/attend_n_excite/txt'
    wordnindex_dir = './prompt_files/attend_n_excite/json/word_n_index/sd1_5x'
    
    
    # arguments for generation
    device = 'cuda:3'
    num_inference_steps  = 50
    use_conform = False # If False, use our self-attn based loss
    update_latent = True # update z_t. if False, original SDv1.5
    seed = 4913
    processor_name = 'processor_3'
    refinement_steps = 20
    if use_conform:
        scale_factor = 1
    else:
        scale_factor = 20 # original value 20
    # processor_name = 'processor_x'
    
    # load prompts and indices
    prompt_list = get_prompt_list_by_line(directory=prompt_dir,file_name=prompt_file_name)
    idx_json = load_json(wordnindex_dir, prompt_file_name)
    
    # save arguments
    save_text_selfattn = False
    save_gen_images = True
    save_crossattn_sim = True
    
    # load model
    model_name = 'sd1_5x'
    # model_name = 'sd1_5x'
    pipe = load_model(model_name=model_name, device=device)
    config = pipe.text_encoder.config
    indices_list = None
    # prompt_list= ['A green glasses and a yellow clock']
    for idx, prompt in enumerate(prompt_list[50:]):
        for i in range(12):
            # to get attn score for each prompt
            pipe.text_encoder.text_model.encoder.layers[i].self_attn.dummy = 0
            
        idx_info_chunk = idx_json[str(idx)][1]
        if prompt_file_name == 'objects':
            indices_list = {
                'obj1':idx_info_chunk['noun1'][0],
                'attr1':idx_info_chunk['adj1'][0],
                'obj2':idx_info_chunk['noun2'][0],
                'attr2':idx_info_chunk['adj2'][0],
            }
        elif prompt_file_name == 'animals':
            indices_list = {
                'obj1':idx_info_chunk['noun1'][0],
                'obj2':idx_info_chunk['noun2'][0],
            }
        elif prompt_file_name == 'animals_objects':
            # breakpoint()
            try:
                if len(idx_info_chunk.keys()) == 2:
                    indices_list = {
                        'obj1':idx_info_chunk['noun1'][0],
                        'obj2':idx_info_chunk['noun2'][0],
                    }
                else:
                    indices_list = {
                        'obj1':idx_info_chunk['noun1'][0],
                        'obj2':idx_info_chunk['noun2'][0],
                        'attr2':idx_info_chunk['adj2'][0],
                    }
            except:
                indices_list = None
                
        if len(idx_json[str(idx)][0].keys()) != 2:
            continue
        else:
            print(prompt)
        # eos_idx = idx_json[str(idx)][1]['noun2'][0] + 1
        eos_idx = 9
        if save_text_selfattn:
            text_sa_save_dir = os.path.join(gen_dir,'text_sa',model_name)
            all_text_maps = pipe.get_text_sa(prompt=prompt, device=device)
            text_sa_save_dir = os.path.join(gen_dir,'text_sa',model_name)
            
            # save each block
            # eos_idx = idx_json[str(idx)][1]['noun2'][0] + 1
            # for i in range(pipe.text_encoder.config.num_hidden_layers):
            #     save_text_sa(text_sa = all_text_maps,
            #                 directory = text_sa_save_dir,
            #                 file_name = f'{prompt}_block_{i}',
            #                 block_to_save = f'block_{i}',
            #                 eos_idx=eos_idx)
                
            # save avg block
            all_text_maps_list = []
            for i in range(pipe.text_encoder.config.num_hidden_layers):
                all_text_maps_list.append(torch.tensor(list(all_text_maps.values())[i]).unsqueeze(0))
            avg = torch.mean(torch.cat(all_text_maps_list, dim=0), dim=0)

            save_text_sa_avg(text_sa = avg,
                            directory = text_sa_save_dir,
                            file_name = f'{prompt}_block_avg',
                            eos_idx=eos_idx)
            
        
        image, all_maps = generate_latent_optimization(
                        prompt = prompt,
                        pipe = pipe,
                        model_name = model_name,
                        processor_name = processor_name,
                        seed = seed,
                        num_inference_steps=num_inference_steps,
                        indices_list=indices_list,
                        use_conform=use_conform,
                        update_latent=update_latent,
                        refinement_steps=refinement_steps,
                        scale_factor=scale_factor
                        )
        if save_crossattn_sim:
            os.makedirs(f"{gen_dir}/{cos_sim_dir}",exist_ok=True)
            cosine_similarity_matrix = torch.zeros(eos_idx-1, eos_idx-1)
            map_save_list = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,49]
            for timestep in map_save_list:
                cross_attn_map = all_maps[0][timestep].reshape(-1, 77)
                for i in range(1,eos_idx):
                    for j in range(1,i+1):
                        cosine_sim = F.cosine_similarity(cross_attn_map[:, i], cross_attn_map[:, j], dim=0)
                        cosine_similarity_matrix[i-1, j-1] = cosine_sim
                
                cosine_similarity_matrix_np = cosine_similarity_matrix.numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(cosine_similarity_matrix_np, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title("Cosine Similarity Matrix")
                plt.savefig(f"{gen_dir}/{cos_sim_dir}/test_{use_conform}_{update_latent}_{prompt}_{timestep}_cossim.png")

        if save_gen_images:
            image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name)
            save_image(image=image[0],directory=image_save_dir, file_name=f'{prompt}_seed_{seed}')

