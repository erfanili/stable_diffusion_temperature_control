from modules.utils_1 import *



data = load_dict(dir = 'generation_outputs', file_name='18:14:36_attn_data_1.pkl')
data = tensors_to_cpu_numpy(data)
data = rearrange_by_layer(data)

arr1 = data['up_2']
# for time in range(20):
#     save_attn_by_layer(attn_array=data['mid'][time,:,:], token = 2, dir = './generation_outputs/mid_data_by_time/1', file_name=f'{time}.jpg')
    
# images_to_gif(dir= 'generation_outputs/mid_data_by_time/1', output_path='generation_outputs/mid_data_by_time/1.gif')


data = load_dict(dir = 'generation_outputs', file_name='18:14:23_attn_data_0.pkl')
data = tensors_to_cpu_numpy(data)
data = rearrange_by_layer(data)


arr0 = data['up_2']

print(arr0-arr1)