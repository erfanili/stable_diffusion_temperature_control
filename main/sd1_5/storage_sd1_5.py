

import torch
from models.attention_processor_x import AttnProcessorX
from utils.tensor_array_utils import *



class AttnFetchSDX():
    def __init__(self,positive_prompt:bool = True):
        self.storage = {}
        self.positive_prompt = positive_prompt
    # def get_timestep_x(self):
    #     timestep = self.unet.timestep.item()
    #     return timestep

    def maps_by_block_x(self):
        """
        final output. organizes data in dicts which keys are blocks and values are numpy arrays of maps in all times
        """

        data = rearrange_by_layers(self.storage)
        data = data_structures_to_cpu_numpy(data)
        
        return data
    
    def store_attn_by_timestep_x(self,time,unet):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        attn_data = self.get_unet_data_x(unet)
        self.storage[time] = attn_data


    def get_unet_data_x(self,unet):
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
    
    
    def set_processor_x(self,unet):
        processors_x = {}
        for layer in unet.attn_processors.keys():
            processor = AttnProcessorX()
            processors_x[layer] = processor
        unet.set_attn_processor(processors_x)