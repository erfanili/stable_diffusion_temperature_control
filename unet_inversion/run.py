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
pipe = load_model(model_class=StableDiffusionPipelineX, model_id = model_id, device = 'cuda:5')
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# print(pipe.scheduler.__class__.__name__)
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

write_to_pickle(dict = output, dir =  './generation_outputs/', file_name = 'attn_data_1.pkl')
save_image(image = image, dir = './generation_outputs', file_name='output_image_1.jpg')
# codes = tokenizer.encode(prompt)
# print([tokenizer.decode(code) for code in codes])
