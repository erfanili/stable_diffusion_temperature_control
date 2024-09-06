import os

# Specify the directory
directory = '/home/erfan/repos/stable_diffusion_temperature_control/main/generation_outputs/sd1_5x/try/attn_dicts'

# List all files in the directory
files = os.listdir(directory)

# Remove file type extension
# file_names = [os.path.splitext(file)[0] for file in files]

# Print the file names
for name in files:
    print(name)