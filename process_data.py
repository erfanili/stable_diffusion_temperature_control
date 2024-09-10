
from utils.load_save_utils import *
from utils.tensor_array_utils import *



process_prompt_files = True

if process_prompt_files:

    txt_directory = './prompt_files/txt/'
    json_words_directory = './prompt_files/json/words/'
    json_idx_directory = './prompt_files/json/indices/'
    file_names = get_file_names(directory=txt_directory)
    for name in file_names:
        obj_words_in_file = {}
        obj_idx_in_file ={}
        prompt_list = get_prompt_list_by_line(directory=txt_directory, file_name=name)
        for idx, prompt in enumerate(prompt_list):
            prompt_words = {}
            prompt_idx ={}
            words = prompt.split(' ')
            index_of_and = words.index('and')
            if words[0] in ('a' , 'an'):
                adj1_idx = 1
                adj1 = words[adj1_idx]
                noun1_idx = list(range(2,index_of_and))   
                noun1 = (' ').join([words[_] for _ in noun1_idx])
                if words[index_of_and+1] in ('a' , 'an'):
                    adj2_idx = index_of_and+2
                    adj2 = words[adj2_idx]
                    noun2_idx = list(range(index_of_and+3,len(words)))
                    noun2 = (' ').join(words[_] for _ in noun2_idx).strip('\n. ')
                    if len(noun2_idx) >1 :
                        print(name,idx)
                else:
                    adj2 = words[index_of_and+1]
                    noun2_idx = list(range(index_of_and+2,len(words)))
                    if len(noun2_idx) >1 :
                        print(name,idx)
                    noun2 = (' ').join(words[_] for _ in noun2_idx).strip('\n. ')
            else:
                adj1 = words[0]     
                noun1 = (' ').join(words[1:index_of_and])
                if words[index_of_and+1] in ('a', 'an'):
                    adj2_idx = index_of_and+1
                    adj2 = words[index_of_and+2]
                    noun2_idx = list(range(index_of_and+3, len(words)))
                    noun2 = (' ').join(words[_] for _ in noun2_idx).strip('\n. ')
                    if len(noun2_idx) >1 :
                        print(name,idx)
                else:
                    adj2_idx = index_of_and+1
                    adj2 = words[adj2_idx]
                    noun2_idx = list(range(index_of_and+2,len(words)))
                    noun2 = (' ').join(words[_] for _ in noun2_idx).strip('\n. ')
                    if len(noun2_idx) >1 :
                        print(name,idx)
                
            prompt_words = {'adj1': adj1, 'noun1': noun1, 'adj2': adj2, 'noun2': noun2}
            prompt_idx = {'adj1': adj1_idx, 'noun1': noun1_idx, 'adj2': adj2_idx, 'noun2': noun2_idx}
            obj_words_in_file[idx] = prompt_words
            obj_idx_in_file[f'obj_{idx}_1'] = noun1_idx
            obj_idx_in_file[f'obj_{idx}_2'] = noun2_idx
        save_json(directory=json_words_directory, file_name=name,data=obj_words_in_file)
        save_json(directory=json_idx_directory, file_name=name,data=obj_idx_in_file)

    
    
    
    
    
    
    




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
