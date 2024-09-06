

import torch
from models.attention_processor_x import AttnProcessorX
from utils.tensor_array_utils import *




class AttnFetchPixartX():
    def __init__(self,positive_prompt:bool = True):
        self.storage = {}
        self.positive_prompt = positive_prompt
    def get_timestep_x(self):
        timestep = self.unet.timestep.item()
        
        return timestep
    
    def maps_by_block_x(self):
        """
        final output. organizes data in dicts which keys are blocks and values are numpy arrays of maps in all times
        """
        data = rearrange_by_layers(self.storage)
        data = data_structures_to_cpu_numpy(data)

        
        return data
    
        
    def store_attn_by_timestep_x(self,time,transformer):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        attn_data = self.get_attn_data_x(transformer)
        self.storage[time] = attn_data
        # print(unet.timestep)



    def get_attn_data_x(self,transformer):
        attn_data = {}

        for i, block in enumerate(transformer.transformer_blocks):
                data = block.attn2.processor.attn_data_x
                attn_data[f'block_{i}'] = data
        
        return attn_data

    def set_processor_x(self,transformer):
        processors_x = {}
        for layer in transformer.attn_processors.keys():
            processor = AttnProcessorX(self.positive_prompt)
            processors_x[layer] = processor
        
        transformer.set_attn_processor(processors_x)