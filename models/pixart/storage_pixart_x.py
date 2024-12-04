

import torch
from typing import Dict
# from models.pixart.t5_attention_x import T5AttentionX
from models.processors import *

# def get_processor_class(processor_name):
#     processor_classes = {
#         'processor_x': AttnProcessorX,
#         'processor_3': AttnProcessor3,
#         'processor_x_2': AttnProcessorX_2,
#         'processor_x_conform': AttnProcessorXCONFORM
# }

    # return processor_classes[processor_name]

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

        
        
class AttnFetchPixartX():
    def __init__(self,positive_prompt:bool = True):
        self.storage = {}
        self.positive_prompt = positive_prompt
    def get_timestep(self):
        timestep = self.unet.timestep.item()
        
        return timestep
    
    def maps_by_block(self):
        """
        final output. organizes data in dicts which keys are blocks and values are numpy arrays of maps in all times
        """
        data = rearrange_by_layers(self.storage)
        data = to_cpu_numpy(data)

        return data
     
    def store_attn_by_timestep(self,time,transformer):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        attn_data = self.get_attn_data(transformer)
        self.storage[time] = attn_data

    def get_attn_data(self,transformer):
        attn_data = {}
        
        for i, block in enumerate(transformer.transformer_blocks):
                data = block.attn2.processor.attn_data_x
                attn_data[f'block_{i}'] = data
        
        return attn_data

    def set_processor(self,transformer,processor_name):
        processors= {}
        for i, layer in enumerate(transformer.attn_processors.keys()):

            processor = get_processor_class(processor_name=processor_name)()
            processors[layer] = processor
        transformer.set_attn_processor(processors)
        
        
        
    #text_encoder maps:
    def store_text_sa(self,text_encoder):
        attn_data = {}
    
        for i, block in enumerate(text_encoder.encoder.block):
                data = block.layer[0].SelfAttention.attn_weights_x
                #avg over heads
                data = torch.mean(data.squeeze(),dim=0)
                attn_data[f'block_{i}'] = data

        return attn_data
    
        
    # def set_text_processor(self, text_encoder):
    #     for block in text_encoder.encoder.block:
    #         block.layer[0].SelfAttention = T5AttentionX(
    #             config=text_encoder.config,
    #             has_relative_attention_bias=False
    #         ).to(text_encoder.device)