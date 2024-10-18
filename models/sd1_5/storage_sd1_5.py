

import torch
from typing import Dict

from models.processors import AttnProcessor3, AttnProcessorX
def to_cpu_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: to_cpu_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_cpu_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu_numpy(item) for item in data)
    else:
        return data
    
    
def rearrange_by_layers(data:dict) -> Dict:
    layers = list(list(data.values())[0].keys())
    rearranged_output = {}
    for layer in layers:
        attn_tensors = torch.stack([data[time][layer] for time in data.keys()])
        rearranged_output[layer] = attn_tensors


    return rearranged_output

class AttnFetchSDX():
    def __init__(self,positive_prompt:bool = True):
        self.storage = {}
        self.positive_prompt = positive_prompt


    def maps_by_block(self):
        """
        final output. organizes data in dicts which keys are blocks and values are numpy arrays of maps in all times
        """

        data = rearrange_by_layers(self.storage)
        data = to_cpu_numpy(data)
        
        return data
    
    def store_attn_by_timestep(self,time,unet):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        attn_data = self.get_unet_data(unet)
        self.storage[time] = attn_data


    def get_unet_data(self,unet):
        """
        saves attention maps in a dict where the keys are unet blocks: down_0 .. down_2, mid, up_1 ... up_3.
        values are attention maps for either positive or negative prompt, averaged over heads, at the current timestep.
        """
        unet_attn_data = {}
        for i0, block in enumerate(unet.down_blocks):
            if block.__class__.__name__ == "CrossAttnDownBlock2D":
                data = block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x
              
                data += block.attentions[1].transformer_blocks[0].attn2.processor.attn_data_x  #[8,4096,77]
 # [4096, prompt_len]
                unet_attn_data[f'down_{i0}'] = data/2
        for i0, block in enumerate(unet.up_blocks):
            if block.__class__.__name__ == "CrossAttnUpBlock2D":
                data = block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x
                data += block.attentions[1].transformer_blocks[0].attn2.processor.attn_data_x #[8,4096,77]# [4096, prompt_len]
                unet_attn_data[f'up_{i0}'] = data/2
        block = unet.mid_block
        data = block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x

        unet_attn_data['mid'] = data
        
        return unet_attn_data
    
    
    def set_processor(self,unet,processor_name, index_data):
        processor_classes = {'processor_x':AttnProcessorX,
                                 'processor_3': AttnProcessor3,}
        processors= {}
        for layer in unet.attn_processors.keys():
            processor = processor_classes[processor_name](idx1 =index_data['obj_1'], idx2 = index_data['obj_2'],eos_idx = index_data['eos'])

            processors[layer] = processor
        unet.set_attn_processor(processors)
        
        
    def store_text_sa(self,text_encoder):
        attn_data = {}
    
        for i, block in enumerate(text_encoder.text_model.encoder.layers):
                data = block.self_attn.attn_data_x
                attn_data[f'block_{i}'] = data

        return attn_data