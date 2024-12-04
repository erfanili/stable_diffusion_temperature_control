import sys
import os
from collections import defaultdict
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# sys.path.append(parent_dir)
import torch
from transformers import T5Tokenizer, T5EncoderModel
from utils.load_save_utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

model_name = "pixart_x"  
pipe = load_model(model_name=model_name, device = 'cuda:0')
tokenizer = pipe.tokenizer
model = pipe.text_encoder

if model_name == 'pixart_x':
    del pipe.transformer
simply_encode = True 

if simply_encode:

    prompt = "a green bench and a red car"

    token_ids = tokenizer(text = prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",)['input_ids']

    text_embeddings = model(token_ids.to(model.device))[0].squeeze().detach()

    cos = nn.CosineSimilarity(dim=0, eps = 1e-6)
    length = 13
    sim = torch.zeros(length, length)
    for i in range(length):
        for j in range(length):
            sim[i,j] = cos(text_embeddings[i],text_embeddings[j])
            
        
    sim = sim.cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(sim, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    plt.savefig(f"{prompt}_{model_name}_cossim.png")



save_attn = False
if save_attn:
    output_dir = './generation_outputs/by_block'
    attn_save_dir = os.path.join(output_dir,'text_attn')
    prompt_list = get_prompt_list_by_line(directory='./',file_name='texture_train')
    for prompt in prompt_list[:100]:

        maps = get_text_attn_map(prompt=prompt,directory = attn_save_dir)
        
        
        
        
get_word_indexes = False
model_name = 'pixart_x'
tokenizer = MyTokenizer(model_name=model_name,device = 'cuda:0')
# tokenizer = pipe.tokenizer
if get_word_indexes:
    directory = './prompt_files/attend_n_excite/txt'
    # file_names = get_file_names(directory=directory)
    # for category in file_names:
    prompt_list = get_prompt_list_by_line(directory = directory, file_name='animals_objects')

    word_n_index = defaultdict(tuple)
    for i,prompt in enumerate(prompt_list):
        word_n_index[i] = attend_n_excite_word_n_index_animals_objects(prompt=prompt, tokenizer = tokenizer)
    save_json(data=word_n_index, directory=f'./prompt_files/attend_n_excite/json/word_n_index/{model_name}', file_name=f'animals_objects')