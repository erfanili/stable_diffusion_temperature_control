import torch
import os
from diffusers import StableDiffusionPipeline
from modules.pipeline_stable_diffusion_x import StableDiffusionPipelineX 
# from modules.unet_2d_condition_x import UNet2DConditionModel
import pickle
from modules.attention_processor_x import AttnProcessorX
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from visualize import write_to_pickle




model_id = "runwayml/stable-diffusion-v1-5" 
pipe = StableDiffusionPipelineX.from_pretrained(model_id, torch_dtype = torch.float32,)
tokenizer = pipe.tokenizer
        
        
pipe.to("cuda:1")
prompt = "an apple on a desk near a window"
image = pipe(prompt = prompt, num_inference_steps = 50 ).images[0]
output = pipe.attn_fetch_x.storage_x

# 

write_to_pickle(dict = output, dir =  './outputs/', file_name = 'attn_data.pkl')
image.save('./outputs/output_image.png')
codes = tokenizer.encode(prompt)
print([tokenizer.decode(code) for code in codes])
