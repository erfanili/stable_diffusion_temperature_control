
from utils.load_save_utils import *
from utils.tensor_array_utils import *



model_name = 'pixart_x'
directory =f'./generation_outputs/{model_name}/try/attn_dicts/'


##if loading all files in directory:
files = get_file_names(directory)

for file_name in files:
    data = load_dict(directory= directory, file_name=file_name)
    print(data.keys())



#### if loading a single file:
file_name = '0_8125'
data = load_dict(directory=directory, file_name=file_name)
print(data.keys())

