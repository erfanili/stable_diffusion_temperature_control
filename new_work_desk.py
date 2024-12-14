from utils.load_save_utils import *

import random
import torch
import os
from models.processors import AttnProcessor3, AttnProcessorX, AttnProcessorX_2
import torch.nn.functional as F
import seaborn as sns
import pickle as pkl

all_models = ['sd1_5', 'sd1_5x', 'pixart', 'pixart_x']
models_with_attn_maps = ['sd1_5x', 'pixart_x']

# processor_classes = {'processor_x': AttnProcessorX,
#                      'processor_3': AttnProcessor3,
#                      'processor_x_2': AttnProcessorX_2,
#                      'processor_avg':AttnProcessor4
#                      }


prompt_files_generate_test = True
if prompt_files_generate_test:
    
  
    prompt_file_name = 'test'
    prompt_dir = './prompt_files'
    device = 'cuda:1'
    num_inference_steps  = 20
    use_conform = False # If False, use our self-attn based loss
    update_latent = False # update z_t. if False, original SDv1.5
    # seed = 4913
    processor_name = 'processor_avg'
    model_name = 'pixart_x'
    gen_dir = f'./generation_outputs/avg_test/original'
    image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name,prompt_file_name)
    attn_map_save_dir = os.path.join(gen_dir,'attn_maps',model_name,processor_name,prompt_file_name)
    text_sa_save_dir = os.path.join(gen_dir,'text_sa',model_name,processor_name,prompt_file_name)
     
    
    prompt_list = get_prompt_list_by_line(directory=prompt_dir,file_name=prompt_file_name)
   
    save_gen_images, save_attn_maps,save_text_selfattn = True, False, False
    pipe = load_model(model_name=model_name, device=device)
    config = pipe.text_encoder.config

    # prompt_loss_dict = {}
    for seed in range(1):
        for idx, prompt in enumerate(prompt_list[0:30]):
            # prompt = 'aldfkjhadslkvjand lakjdn alkds hnalksdjn alksdj nalkds nalkdjnalksd jnalskd jnaslkd nalksdjn alskdjn'
            # for i in range(12):
                # to get attn score for each prompt
                # pipe.text_encoder.text_model.encoder.layers[i].self_attn.dummy = 0
                
            
            image, all_maps = generate(
                            prompt = prompt,
                            pipe = pipe,
                            model_name = model_name,
                            processor_name = processor_name,
                            index_data = {},
                            seed = seed,
                            num_inference_steps=num_inference_steps,
                            blocks_to_save = ['block_13']
                            )

            all_text_maps = pipe.attn_fetch_x.store_text_sa(text_encoder = pipe.text_encoder)
            if save_gen_images:
                save_image(image=image, directory=image_save_dir, file_name=f'{idx}_{prompt}_seed_{seed}')
            if save_attn_maps:
                save_pkl(directory=attn_map_save_dir,data=all_maps, file_name = f'{idx}_{seed}')
                # data =np.mean(np.mean(all_maps['up_1'],axis=0),axis=0)
                # np.set_printoptions(precision=1, suppress=True)
                # print(data[:7,:7])
    
            #avg over blocks
            if save_text_selfattn:
                
                # avg = torch.mean(torch.stack([maps for maps in all_text_maps.values()], dim=0), dim=0)
                avg =all_text_maps['block_6'][4]
                np.set_printoptions(precision=2)

                # print(np.mean(avg.detach().cpu().numpy(),axis=0)[:7,:7])
        
                print(avg.detach().cpu().numpy()[:7,:7])
                save_pkl(data = avg,directory=text_sa_save_dir,file_name = f'{idx}_{prompt}')

    print(f"complete {gen_dir}")


save_text = False 
if save_text:
    prompt_file_name = 'color_train'
    prompt_dir = './prompt_files/comp_bench/txt'
    device = 'cuda:3'
    num_inference_steps  = 50
    use_conform = False # If False, use our self-attn based loss
    update_latent = False # update z_t. if False, original SDv1.5
    # seed = 4913
    processor_name = 'processor_avg'
    model_name = 'pixart'
    gen_dir = f'./generation_outputs/'
    image_save_dir = os.path.join(gen_dir,'images',model_name,processor_name)
    attn_map_save_dir = os.path.join(gen_dir,'attn_maps',model_name,processor_name)
     
    
    prompt_list = get_prompt_list_by_line(directory=prompt_dir,file_name=prompt_file_name)
    gen_dir = f'./generation_outputs/s_t_optimization/{prompt_file_name}'
    text_sa_save_dir = os.path.join(gen_dir,'text_sa',model_name,processor_name)
    # sa_sav
    # e_dir = os.path.join(gen_dir,'sa',model_name,processor_name)
    # attn_map_save_dir = os.path.join(gen_dir,'attn_maps',model_name,processor_name)
    pipe = load_model(model_name=model_name, device=device)
    config = pipe.text_encoder.config
    
    # for seed in range(3):
     
    text_sa_data = {}
    for idx, prompt in enumerate(prompt_list):
        for i in range(12):
            # to get attn score for each prompt
            pipe.text_encoder.text_model.encoder.layers[i].self_attn.dummy = 0
        
            # all_text_maps, eos_idx, tokenized_text = pipe.get_text_sa(prompt=prompt, device=device)
            # text_sa_save_dir = os.path.join(gen_dir,'text_sa',model_name)
            
# def get_text_sa(self, prompt, device, num_images_per_prompt=1, negative_prompt=None, lora_scale=None):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt='',
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        # return self.attn_fetch_x.store_text_sa(text_encoder = self.text_encoder), prompt_embeds
        # return prompt_embeds
        all_text_maps = pipe.attn_fetch_x.store_text_sa(text_encoder = pipe.text_encoder)
          
        # avg_map = np.mean(np.stack([map_i for map_i in all_text_maps.values()],axis = 0),axis = 0)
          
        # all_text_maps_list = []
        # for i in range(pipe.text_encoder.config.num_hidden_layers):
  
        avg = torch.mean(torch.cat([map_i.unsqueeze(0) for map_i in all_text_maps.values()], dim=0), dim=0)
        print(torch.mean(avg,axis=0)[:7,:7])
        
        avg = avg.detach()[:10,:10]
        # save_text_sa_avg(avg,
        #          directory='./',
        #          file_name='sa',
        #          eos_idx=None)
        
        # exit()
        text_sa_data[idx] = avg
    save_pkl(data = text_sa_data,directory=text_sa_save_dir,file_name = 'color_train')
    
False
new_test_text = False 
if new_test_text:
    data = load_pkl(directory='/home/erfan/repos/stable_diffusion_temperature_control/generation_outputs/R/attn_maps/sd1_5x/processor_x/compbench_color_train', file_name='1_2')
    data =np.mean(np.mean(data['up_1'],axis=0),axis=0)
    np.set_printoptions(precision=1, suppress=True)
    print(data[:7,:7])
    
    

map_hist = False
if map_hist:
    directory='/home/erfan/repos/stable_diffusion_temperature_control/generation_outputs/self_attn_hist/attn_maps/sd1_5x/processor_x/compbench_color_train'
    data = load_pkl(directory=directory, file_name='0_0')
    data = data['up_1']
    data = data.reshape(50,-1)
    for idx in range(len(data)):
        plt.hist(data[idx],bins = 100)
        plt.yscale('log')
        plt.savefig(f'{directory}/time_{idx}.png')
        plt.close()
    