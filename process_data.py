
from utils.load_save_utils import *
from utils.tensor_array_utils import *



process_prompt_files = True

if process_prompt_files:

    txt_directory = './prompt_files/txt/'
    json_directory = './prompt_files/json/'
    file_names = get_file_names(directory=txt_directory)
    for name in file_names:
        file_data = {}
        prompt_list = get_prompt_list_by_line(directory=txt_directory, file_name=name)
        for idx, prompt in enumerate(prompt_list):
            words = prompt.split(' ')
            if 'and' not in words:
                print(name, idx)
            else:
                index_of_and = words.index('and')
                if words[0] in ('a' , 'an'):
                    adj1 = words[1]     
                    noun1 = (' ').join(words[2:index_of_and])
                    if words[index_of_and+1] in ('a' , 'an'):
                        adj2 = words[index_of_and+2]
                        noun2 = (' ').join(words[index_of_and+3:]).strip('\n. ')
                    else:
                        adj2 = words[index_of_and+1]
                        noun2 = (' ').join(words[index_of_and+2:]).strip('\n. ')
                else:
                    adj1 = words[0]     
                    noun1 = (' ').join(words[1:index_of_and])
                    if words[index_of_and+1] in ('a', 'an'):
                        adj2 = words[index_of_and+2]
                        noun2 = (' ').join(words[index_of_and+3:]).strip('\n. ')
                    else:
                        adj2 = words[index_of_and+1]
                        noun2 = (' ').join(words[index_of_and+2:]).strip('\n. ')
                    
                prompt_data = {'adj1': adj1, 'noun1': noun1, 'adj2': adj2, 'noun2': noun2}
            file_data[idx] = prompt_data
        save_json(directory=json_directory, file_name=name,data=file_data)
    
    
    
    
    
    
    
    




process_attn_data = False
if process_attn_data:

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
