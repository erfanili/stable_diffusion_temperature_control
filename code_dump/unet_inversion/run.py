
import torch
import os
from diffusers import StableDiffusionPipeline
from modules.pipeline_stable_diffusion_x import StableDiffusionPipelineX
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from modules.unet_2d_condition_x import UNet2DConditionModel
import pickle
from modules.attention_processor_x import AttnProcessorX
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from modules.utils_1 import *
from modules.utils import *
import random

model_id = "runwayml/stable-diffusion-v1-5"
model_class = StableDiffusionPipelineX
device = 'cuda:0'


pipe = load_model(model_class=model_class, model_id = model_id, device = device)
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.config.timestep_spacing = "linspace"
# pipe = StableDiffusionPipelineX.from_pretrained(model_id, torch_dtype = torch.float32, scheduler =scheduler)
# tokenizer = pipe.tokenizer

        
# pipe.to("cuda:1")
prompt = "an apple on a desk near a window"



seeds = [random.randint(0,1000000) for _ in range(1)]
seeds = [42]
with torch.no_grad():
    for seed in seeds:
        generator = torch.manual_seed(seed)
        image = pipe(prompt = prompt, num_inference_steps = 20 , generator = generator).images[0]
        output = pipe.attn_fetch_x.storage_x

# # 

write_to_pickle(dict = output, dir =  './generation_outputs/', file_name = 'negative.pkl')
save_image(image = image, dir = './generation_outputs', file_name='output_image.jpg')
# codes = tokenizer.encode(prompt)
# print([tokenizer.decode(code) for code in codes])
