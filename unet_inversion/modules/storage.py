

import torch
from modules.attention_processor_x import AttnProcessorX




class AttnFetchX():
    def __init__(self):
        self.storage_x = {}

    def get_timestep_x(self):
        timestep = self.unet.timestep.item()
        
        return timestep
    
    
    def get_unet_data(self):
        
        return self.unet.attn_data
        
    def store_attn_by_timestep_x(self,time,unet):
        

        attn_data = self.get_unet_data_x(unet)
        self.storage_x[time] = attn_data
        # print(unet.timestep)

    
    def set_processor_x(self,unet):

        processors_x = {}
        for layer in unet.attn_processors.keys():
            processor = AttnProcessorX()
            processors_x[layer] = processor
        
        unet.set_attn_processor(processors_x)

    def get_unet_data_x(self,unet):
        unet_attn_data = {}
        with torch.no_grad():
            for i0, block in enumerate(unet.down_blocks):
                if block.__class__.__name__ == "CrossAttnDownBlock2D":
                    b0 = block.attentions[0].transformer_blocks[0]
                    b1 = block.attentions[1].transformer_blocks[0]
                    data00 = b0.attn2.processor.attn_data_x[0]
                    data01 = b0.attn2.processor.attn_data_x[1]
                    data10 = b1.attn2.processor.attn_data_x[0]
                    data11 = b1.attn2.processor.attn_data_x[1]
                    


                    data_cat = torch.cat((data00.unsqueeze(0),data10.unsqueeze(0)), dim = 0)
                    data_mean = torch.mean(data_cat,dim=0)

                    unet_attn_data[f'down_{i0}'] = data_mean
            
            for i0, block in enumerate(unet.up_blocks):
                if block.__class__.__name__ == "CrossAttnUpBlock2D":
                    data1 = block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x
                    data2 = block.attentions[1].transformer_blocks[0].attn2.processor.attn_data_x
                    print(data2.size())
                    data_cat = torch.cat((data1,data2), dim = 0)
                    data_mean = torch.mean(data_cat,dim=0)

                    unet_attn_data[f'up_{i0}'] = data_mean
                    
            block = unet.mid_block
            data1 = block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x
            data_cat = data1
            print(data1.size())
            data_mean = torch.mean(data_cat,dim=0)

            unet_attn_data['mid'] = data_mean
                   
        return unet_attn_data