import pickle as pkl 

with open('/home/erfan/repos/stable_diffusion_temperature_control/evaluation_codes/color_train_test_labels.pkl', 'rb') as f:
    labels = pkl.load(f)
    print(labels)