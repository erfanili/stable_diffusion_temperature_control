

from utils.load_save_utils import *
import matplotlib.pyplot as plt

import scipy.stats as st



# breakpoint()

def get_pos_neg_classes(dataset:dict):
    positives = []
    negatives = []
    num_data = len(dataset['label'])
    for idx in range(num_data):
        if dataset['label'][idx] == '0':
            negatives.append(dataset['input'][idx])
        elif dataset['label'][idx] == '1':
            positives.append(dataset['input'][idx])
    return positives, negatives

def normalize(array_list:list)->list:
    
    def center(attn_map):
        attn_map = attn_map.astype(np.float128)
        return (attn_map - np.mean(attn_map)*np.ones_like(attn_map))/(np.std(attn_map)+1e-10)
    new_list = []
    for all_times_map in array_list:
        normed_array =  np.stack([center(attn_map) for attn_map in all_times_map])
        new_list.append(normed_array)    
    return new_list

def get_stats(array_list:list,stat:str)->list:
    if stat not in ('mean', 'std', 'skew', 'kurt'):
        return print('give a valid stat')
    
    else:
        results_list = []
        for array in array_list:
            array = array.astype(np.float128)
            if stat == 'mean':
                result = np.mean(array)
            elif stat == 'std':
                result = np.std(array) 
            elif stat == 'skew':
                result = st.skew(array, axis = None)
            elif stat == 'kurt':
                result = st.kurtosis(array, axis = None)
            
            results_list.append(result)
            
    return results_list


def save_histogram(array:np.array, directory:str, file_name:str,bins = 200):
    os.makedirs(directory, exist_ok=True)
    
    file_path = os.path.join(directory,file_name)
    plt.hist(array.flatten(),bins=bins)
    plt.savefig(fname=f'{file_path}.jpg')
    plt.close()


def plot_stats(maps,stat,directory,file_name):
    stat_dir =os.path.join(directory,stat)
    os.makedirs(stat_dir,exist_ok=True)
    file_path = os.path.join(stat_dir,file_name)
    maps_by_time =[array for array in maps]
    stats = get_stats(array_list=maps_by_time,stat = stat)
    plt.plot(np.arange(len(stats)),stats)
    plt.savefig(f'{file_path}.jpg')
    plt.close()
    

make_pos_neg = True
if make_pos_neg:
    dataset_dir = './dataset'
    target_category = 'color_train'
    dataset = load_pkl(directory=dataset_dir,file_name=f'{target_category}_pixart_512_block_13_time_all')
    positives,negatives = get_pos_neg_classes(dataset)
    normed_positives = normalize(array_list=positives[:50])
    normed_negatives = normalize(array_list=negatives[:50])

    
    for idx in range(50):
        for stat in ['mean', 'std', 'skew', 'kurt']:
            plot_stats(maps = normed_positives[idx],directory='./plots/',stat = stat, file_name=f'{idx}')
    # for idx,array in enumerate(normed_positives[:50]):
    #     save_histogram(array=array, directory = './histograms/positives',file_name= f'{idx}')

    # for idx,array in enumerate(normed_negatives[:50]):
    #     save_histogram(array=array, directory = './histograms/negatives',file_name= f'{idx}')





# stat = 'kurt'
# positive_means_avg = np.mean(np.array(get_avg_stat(array_list = normed_positives,stat = stat)))
# negative_means_avg = np.mean(np.stack(get_avg_stat(array_list = normed_negatives,stat = stat)))

# # print(normed_positives)
# print(positive_means_avg, negative_means_avg)
