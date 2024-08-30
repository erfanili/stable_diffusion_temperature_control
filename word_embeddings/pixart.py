from utils import *

from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline
import torch
import gc
import random


prompt1 = "a dog on the street"
prompt2 = "a cat on the street"


seeds = [random.randint(0,1000000) for _ in range(10)]


for seed in seeds:
    generator = torch.manual_seed(seed)
        
        
    embeds1 =pixart_generate_embeds(prompt1,device = 'cuda:4')
    image1 = pixart_generate_image(embeds1,device = 'cuda:4',generator = generator,num_inference_steps = 20)
    embeds2 =pixart_generate_embeds(prompt2,device = 'cuda:4')
    image2 = pixart_generate_image(embeds2,device = 'cuda:4', generator = generator,num_inference_steps = 20)
    save_image(image = image1, dir = f'./generation_outputs/pixar_cat_dog_whole', file_name = f'{seed}_1')
    save_image(image = image2, dir = f'./generation_outputs/pixar_cat_dog_whole', file_name = f'{seed}_2')
    
    embeds_x = embeds1
    embeds_x[0] = (embeds1[0]+embeds2[0])/2
    embeds_x[2] = (embeds1[2]+embeds2[2])/2
    
    image2 = pixart_generate_image(embeds2,device = 'cuda:4', generator = generator,num_inference_steps = 20)
    save_image(image = image2, dir = f'./generation_outputs/pixar_cat_dog_whole', file_name = f'{seed}_x')