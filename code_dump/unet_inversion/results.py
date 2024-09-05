from modules.utils_1 import *



data = load_dict(dir = 'generation_outputs', file_name='22:32:19_positive.pkl')
data = tensors_to_cpu_numpy(data)
data = rearrange_by_layer(data)


for time in range(20):
    save_attn_by_layer(attn_array=data['up_1'][time,:,:], token = 2, dir = './generation_outputs/up_1_data_by_time/positive', file_name=f'{time}.jpg')
    
images_to_gif(dir= 'generation_outputs/up_1_data_by_time/positive', output_path='generation_outputs/up_1_data_by_time/positive.gif')


data = load_dict(dir = 'generation_outputs', file_name='22:32:41_negative.pkl')
data = tensors_to_cpu_numpy(data)
data = rearrange_by_layer(data)

for time in range(20):
    save_attn_by_layer(attn_array=data['up_1'][time,:,:], token = 2, dir = './generation_outputs/up_1_data_by_time/negative', file_name=f'{time}.jpg')
    
images_to_gif(dir= 'generation_outputs/up_1_data_by_time/negative', output_path='generation_outputs/up_1_data_by_time/negative.gif')


