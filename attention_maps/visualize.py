import pickle 
import os
import torch
import numpy as np 
import math
from typing import Dict
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
    
    

def write_to_pickle(dict: dict,dir: str, file_name: str):
    os.makedirs(dir, exist_ok = True)
    file_path = os.path.join(dir,file_name)
    with open (f'{file_path}', 'wb') as f:
        pickle.dump(dict, f)



def images_to_gif(dir, output_path, duration=200, loop=1000):
    os.makedirs(dir, exist_ok=True)
    image_paths = os.listdir(dir)
    num_timesteps = len(image_paths)
    sorted_image_paths = [os.path.join(dir,f'{i}.jpg') for i in range(num_timesteps)]
    images = [Image.open(image) for image in sorted_image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],  # Add the rest of the images
        duration=duration,          # Duration for each frame (in milliseconds)
        loop=loop                   # Number of loops (0 for infinite loop)
    )


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



if __name__ == "__main__":
    data = load_dict(dir= './outputs', file_name='attn_data.pkl')
    data = tensors_to_cpu_numpy(data)
    data = rearrange_by_layer(data)
    # data = get_average_over_time(data)
    write_to_pickle(dict = data, dir='./outputs', file_name= 'attn_data_arr_avg.pkl')
    for layer in data.keys():
        num_timesteps = len(data[layer])
        for time in range(num_timesteps):    # print(data[layer].shape)
            save_attn_by_layer(data[layer][time], token = 2, output_dir=f'./outputs/attn_by_layer/{layer}/',output_name= f'{time}.jpg')
    # attn_sum = total_attn_by_token(data)
    # print(attn_sum)
    images_to_gif(dir = './outputs/attn_by_layer/up_1/', output_path= './outputs/up_1.gif')