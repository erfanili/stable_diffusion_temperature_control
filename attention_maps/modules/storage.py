from collections import defaultdict




class AttnFetchX():
    def __init__(self,unet):
        self.unet = unet
        self.storage_x = {}

    def get_timestep_x(self):
        timestep = self.unet.timestep.item()
        
        return timestep
    
    
    def get_unet_data(self):
        
        return self.unet.attn_data
        
    def store_attn_by_timestep_x(self,time):
        

        attn_data = self.get_unet_data_x()
        self.storage_x[time] = attn_data


    def get_unet_data_x(self):
        unet_attn_data = []
        for i0, block in enumerate(self.unet.down_blocks):
            # print(i0,block.__class__.__name__)
            if block.__class__.__name__ == "CrossAttnDownBlock2D":
                for i1, transformer_model in enumerate(block.attentions):
                    for i2, transformer_block in enumerate(transformer_model.transformer_blocks):
                        attn_1 = transformer_block.attn1
                        layer_1_address = (i0,i1,i2,1)
                        attn_2 = transformer_block.attn2
                        layer_2_address = (i0,i1,i2,2)
                        unet_attn_data.append((layer_1_address,  attn_1.processor.attn_data_x))
                        unet_attn_data.append((layer_2_address , attn_2.processor.attn_data_x))
                        
                    
        return unet_attn_data