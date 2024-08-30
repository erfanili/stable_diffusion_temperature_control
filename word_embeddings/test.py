import torch 
import os 
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline 
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch.nn.functional as F
from datetime import datetime 
from utils import load_model, ensemble_embed, token_wise_ensemble_embed, encode_prompt



if __name__ == "__main__":
    pipe = load_model(model_to_load='sd_pipeline', device='cuda:4')
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    main_prompt = "a photo of an apple macbook on a desk near a window with a stack of books and a couple of chairs and curtains and a woman sitting on the chair and fruits on the desk."
    
    prompt_list =[
    "On a desk by a window, there's a MacBook, some fruits, a stack of books, and a woman sitting on a nearby chair with curtains hanging in the background.",
    "A woman sits on a chair near a window, with curtains behind her, a MacBook on the desk, along with a stack of books and some fruits.",
    "Near the window, where curtains hang, a desk holds a MacBook, fruits, and a stack of books, with a woman seated on a nearby chair.",
    "A MacBook sits on a desk next to some fruits and a stack of books, with a woman on a chair near a window with curtains.",
    "By the window with curtains, a desk holds a MacBook, books, and fruits, while a woman is seated on a nearby chair.",
    "A stack of books, a MacBook, and fruits are on a desk near a window, with a woman sitting on a chair and curtains in the background.",
    "There's a MacBook on a desk with fruits and books, close to a window where a woman sits on a chair near curtains.",
    "On a desk by a curtained window, a MacBook rests beside fruits and a stack of books, while a woman sits on a nearby chair.",
    "A woman is seated on a chair near a window with curtains, with a MacBook, books, and fruits on the nearby desk.",
    "By a window with curtains, a woman sits on a chair while a desk beside her holds a MacBook, a stack of books, and some fruits.",
    "A woman sits near a window with curtains, with a desk beside her holding a MacBook, books, and fruits.",
    "Near a window with curtains, a desk holds a MacBook, a stack of books, and fruits, with a woman sitting on a chair nearby.",
    "A woman is seated by a window with curtains, with a MacBook, fruits, and a stack of books on the desk next to her.",
    "On a desk near a window, where a woman sits on a chair, a MacBook is placed beside some fruits and a stack of books, with curtains in the background.",
    "A MacBook, a stack of books, and fruits sit on a desk by a window with curtains, with a woman sitting on a nearby chair.",
    "A woman sits near a window with curtains, and on the desk beside her are a MacBook, books, and fruits.",
    "A desk near a window with curtains holds a MacBook, books, and fruits, while a woman sits on a nearby chair.",
    "There is a MacBook on a desk with fruits and books, next to a window with curtains, and a woman sitting on a nearby chair.",
    "A woman sits on a chair by a window with curtains, with a desk nearby holding a MacBook, books, and fruits.",
    "On a desk near a curtained window, there's a MacBook, a stack of books, and some fruits, with a woman seated on a chair nearby."
]
    new_embed = token_wise_ensemble_embed(main_prompt=main_prompt ,aux_prompt_list = prompt_list, tokenizer = tokenizer, text_encoder = text_encoder)
    
    