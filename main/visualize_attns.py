





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


