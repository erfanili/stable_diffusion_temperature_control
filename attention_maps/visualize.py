import pickle 
import os
import torch
import numpy as np 
import math
from PIL import Image
def write_to_pickle(dict: dict,dir: str, file_name: str):
    os.makedirs(dir, exist_ok = True)
    file_path = os.path.join(dir,file_name)
    with open (f'{file_path}', 'wb') as f:
        pickle.dump(dict, f)
        

def load_dict(dir:str, file_name:str):
    
    file_path = os.path.join(dir,file_name)
    if not os.path.exists(file_path):
        
        print("File does not exist.")
        
    else:
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        return data
    





def tensors_to_cpu_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: tensors_to_cpu_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [tensors_to_cpu_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(tensors_to_cpu_numpy(item) for item in data)
    elif isinstance(data, set):
        return {tensors_to_cpu_numpy(item) for item in data}
    else:
        return data
    
    
    
def get_average_attn_over_time(data):
    layers = ['down_0','down_1','down_2','mid','up_1','up_2','up_3']
    output = {}
    for layer in layers:
        
        attn_arrays = np.array([data[time][layer] for time in data.keys()])
        output[layer] = np.mean(attn_arrays,axis = 0)

    return output

def write_to_pickle(dict: dict,dir: str, file_name: str):
    os.makedirs(dir, exist_ok = True)
    file_path = os.path.join(dir,file_name)
    with open (f'{file_path}', 'wb') as f:
        pickle.dump(dict, f)




def visualize_attn(attn_array, token, output_dir, output_name):
    #rescale values to get value 128 for patches on average
    clip_seq_length = 77
    attn = attn_array[:,token]*255* clip_seq_length/2
    size = int(math.sqrt(len(attn)))
    attn = attn.reshape((size,size))
    # print(np.sum(attn))
    attn = attn.astype(np.uint16)
    
    os.makedirs(output_dir, exist_ok = True)
    file_path = os.path.join(output_dir,output_name)
    im = Image.fromarray(attn)
    im.save(file_path)
    
if __name__ == "__main__":
    
    data = load_dict(dir= './outputs', file_name='attn_data.pkl')
    data = tensors_to_cpu_numpy(data)
    data = get_average_attn_over_time(data)
    write_to_pickle(dict = data, dir='./outputs', file_name= 'attn_data_arr_avg.pkl')
    for layer in data.keys():
        visualize_attn(data[layer], token = 2, output_dir='./outputs/attn_by_layer',output_name= f'{layer}.png')
    print(data)