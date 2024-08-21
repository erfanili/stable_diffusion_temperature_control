import pickle 




def load_dict(file_path:str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    return data 




data = load_dict('./outputs/attn_data2.pkl')
attn_data = list(data.values())
print(data)
def get_average_attn_by_time(data):
    avg_attn_by_time = {}
    for time in data.keys():
        attns_at_this_time = data[time]