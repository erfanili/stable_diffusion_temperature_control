from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPPreTrainedModel, CLIPModel, CLIPMLP
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers import PixArtAlphaPipeline
from transformers import T5EncoderModel
from transformers import BertTokenizer, BertModel
from scipy import spatial
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from collections import defaultdict
import gc
import numpy as np
from datetime import datetime 
import os
import pickle


import pickle 
import os
import torch
import numpy as np 
import math
from typing import Dict
from PIL import Image

import pickle 
import os
import torch
import numpy as np 
import math
from typing import Dict
from PIL import Image






def resahpe_n_scale_array(array,size):
    array = (array - np.min(array))/(np.max(array) -np.min(array))*255
    array = array.reshape((size,size)).astype(np.uint16)
    return array


def save_attn_by_layer(attn_array, token, output_dir, output_name):
    attn = attn_array[:,token]
    size = int(math.sqrt(len(attn)))
    attn = resahpe_n_scale_array(attn,size)
    print(attn.shape)

    os.makedirs(output_dir, exist_ok = True)
    file_path = os.path.join(output_dir,output_name)
    im = Image.fromarray(attn)
    im = im.resize((256,256))
    if im.mode == 'I;16':
        im = im.convert('L')
    im.save(file_path)
    
    
    
def rearrange_by_layer(data:dict) -> Dict:
    layers = ['down_0','down_1','down_2','mid','up_1','up_2','up_3']
    rearranged_output = {}
    for layer in layers:
        attn_arrays = np.array([data[time][layer] for time in data.keys()])
        rearranged_output[layer] = attn_arrays
    def get_average_over_heads(data) -> Dict:
        avg = {layer: np.mean(data[layer],axis = 1) for layer in data.keys()}    
        return avg 
    rearranged_head_avg = get_average_over_heads(rearranged_output)
    return rearranged_head_avg


    
def total_attn_by_token(data:dict):
    output = {}
    for layer in data.keys():
        array = data[layer]
        output[layer] =  np.sum(array,axis = 0).astype(np.uint16)
    return output



def get_average_over_time(data) -> Dict:
    output = {}
    for layer in data.keys():
        output[layer] = np.mean(data[layer],axis = 0)
    return output



def get_last_attn(data) -> Dict:
    output = {}
    for layer in data.keys():
        output[layer] = data[layer][-1]
    return output




    
def rearrange_by_layer(data:dict) -> Dict:
    layers = ['down_0','down_1','down_2','mid','up_1','up_2','up_3']
    rearranged_output = {}
    for layer in layers:
        attn_arrays = np.array([data[time][layer] for time in data.keys()])
        rearranged_output[layer] = attn_arrays
    def get_average_over_heads(data) -> Dict:
        avg = {layer: np.mean(data[layer],axis = 1) for layer in data.keys()}    
        return avg 
    rearranged_head_avg = get_average_over_heads(rearranged_output)
    return rearranged_head_avg


    
def total_attn_by_token(data:dict):
    output = {}
    for layer in data.keys():
        array = data[layer]
        output[layer] =  np.sum(array,axis = 0).astype(np.uint16)
    return output



def get_average_over_time(data) -> Dict:
    output = {}
    for layer in data.keys():
        output[layer] = np.mean(data[layer],axis = 0)
    return output



def get_last_attn(data) -> Dict:
    output = {}
    for layer in data.keys():
        output[layer] = data[layer][-1]
    return output






def ensemble_embed(prompt_list: list,tokenizer,text_encoder) -> torch.Tensor:
    embedding_list = []
    for prompt in prompt_list:
        prompt_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['input_ids']
        encoded_prompt = text_encoder(prompt_ids.to(text_encoder.device)).last_hidden_state
        embedding_list.append(encoded_prompt)
        
    embedding_tuple = tuple(embedding_list)
    ensemble_tensor = torch.cat(embedding_tuple,dim=0)
    mean_teansor = torch.mean(ensemble_tensor,dim=0)
    return mean_teansor.unsqueeze(0)


def encode_prompt_by_position(prompt:str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel)-> Dict:
    prompt_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['input_ids']
    prompt_ids = prompt_ids.to(text_encoder.device)
    encoded_prompt = text_encoder(prompt_ids).last_hidden_state.squeeze()
    output = {position: (prompt_ids.squeeze()[position].item() , encoded_prompt[position]) for position in range(77)}

    return output 




def token_wise_ensemble_embed(main_prompt: str,aux_prompt_list: list,tokenizer,text_encoder) -> torch.Tensor:
    main_embed = encode_prompt_by_position(prompt= main_prompt, tokenizer=tokenizer, text_encoder=text_encoder)
    storage = []
    new_embed = []
    for position in main_embed.keys():
        id = main_embed[position][0]
        embed = main_embed[position][1]
        for aux_prompt in aux_prompt_list:
            aux_embed = encode_prompt_by_position(prompt=aux_prompt, tokenizer=tokenizer, text_encoder=text_encoder)
            for aux_token_embed in aux_embed.values():
                if  (id == aux_token_embed[0] and id !=49407):
                    storage.append(aux_token_embed[1].unsqueeze(0).detach())
        mean_tensor = torch.mean(torch.cat(tuple(storage),dim = 0),dim =0)
        embed += 0.1*mean_tensor
        new_embed.append(embed.unsqueeze(0))
    new_embed = torch.cat(tuple(new_embed))
    
    return new_embed.unsqueeze(0)
    
    


def encode_prompt_by_id(prompt:str, tokenizer, text_encoder)-> Dict:
    prompt_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['input_ids']
    encoded_prompt = text_encoder(prompt_ids.to(text_encoder.device)).last_hidden_state.squeeze()
    
    return {prompt_ids.squeeze()[position].item() : encoded_prompt[position] for position in range(77)}


def simply_encode_prompt(prompt:str, tokenizer, text_encoder)-> Dict:
    input_data = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['input_ids']
    encoded_prompt = text_encoder(input_data.to(text_encoder.device)).last_hidden_state.squeeze()
    
    return encoded_prompt



def get_embedding_by_token(token,prompt):
    prompt_embedding = encode_prompt_by_id(prompt)
    token_id = tokenizer(
                token,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['input_ids'].squeeze()[1].item()
    if token_id in prompt_embedding.keys():
        token_embedding = prompt_embedding[token_id]
    
        return token_embedding.detach().cpu().numpy()
    else:
        
        return None
    
    
def get_embedding_by_position(position:int,prompt):
    prompt_embedding = simply_encode_prompt(prompt).squeeze()
    embedding = prompt_embedding[position]
    
    return embedding.detach().cpu().numpy()




def get_distances(position, main_prompt, prompt_list):
    distances = []
    main_embedding = get_embedding_by_position(position, prompt = main_prompt)
    for prompt in prompt_list:
        aux_embedding = get_embedding_by_position(position, prompt=prompt)
        if aux_embedding is not None:
            distance = spatial.distance.cosine(main_embedding,aux_embedding)
            distances.append(distance)
    
    
    
    dist_arr = np.array(distances)
    if len(dist_arr) > 0 :
        return f'mean: {np.mean(dist_arr):.2f}, std: {np.std(dist_arr):.2f}'
    else:
        return 'no occurances'


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)




def pixart_generate_embeds(prompt:str, device):

    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        transformer=None, device_map = 'balanced'
    )

    with torch.no_grad():
        prompt = prompt
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
        gc.collect()
        torch.cuda.empty_cache()
        del pipe
    
    return [prompt_embeds.to(device), prompt_attention_mask.to(device), negative_embeds.to(device), negative_prompt_attention_mask.to(device)]



def pixart_generate_image(embeds:tuple,device,generator,num_inference_steps = 20):
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",text_encoder=None,
    ).to(device)
    prompt_embeds = embeds[0]
    prompt_attention_mask = embeds[1]
    negative_embeds = embeds[2]
    negative_prompt_attention_mask = embeds[3]
    latents = pipe(
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        num_images_per_prompt=1,
        output_type="latent",
        generator = generator,
        num_inference_steps = num_inference_steps
    ).images

    del pipe.transformer
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    return image
    









# class CLIPCrossAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.embed_dim = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.embed_dim // self.num_heads
#         self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
#         self.mlp = CLIPMLP(config)
#         self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
#         if self.head_dim * self.num_heads != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
#                 f" {self.num_heads})."
#             )
#         self.scale = self.head_dim**-0.5
#         self.dropout = config.attention_dropout

#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

#     def cross_attn(
#         self,
#         main_prompt_embeds: torch.Tensor,
#         aux_prompt_embeds: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         causal_attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """Input shape: Batch x Time x Channel"""
        
#         bsz, tgt_len, embed_dim = main_prompt_embeds.size()

#         # get query proj
#         query_states = self.q_proj(aux_prompt_embeds) * self.scale
#         key_states = self._shape(self.k_proj(main_prompt_embeds), -1, bsz)
#         value_states = self._shape(self.v_proj(main_prompt_embeds), -1, bsz)

#         proj_shape = (bsz * self.num_heads, -1, self.head_dim)
#         query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
#         key_states = key_states.view(*proj_shape)
#         value_states = value_states.view(*proj_shape)

#         src_len = key_states.size(1)
#         attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

#         if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         # apply the causal_attention_mask first
#         if causal_attention_mask is not None:
#             if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
#                     f" {causal_attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         if output_attentions:
#             # this operation is a bit akward, but it's required to
#             # make sure that attn_weights keeps its gradient.
#             # In order to do so, attn_weights have to reshaped
#             # twice and have to be reused in the following
#             attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
#         else:
#             attn_weights_reshaped = None

#         attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

#         attn_output = torch.bmm(attn_probs, value_states)

#         if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

#         hidden_states = self.out_proj(attn_output)
        
        
#         return hidden_states
    
#     def forward(self,hidden_states, aux_states):
#         residual = hidden_states
#         hidden_states = self.layer_norm1(hidden_states)
#         hidden_states = self.cross_attn(hidden_states,aux_states)
        
        
#         hidden_states = residual + hidden_states

#         residual = hidden_states
#         hidden_states = self.layer_norm2(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         return hidden_states

