import os
import glob

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'input')

n_classes = 10

locations = list(set(f.split('/')[-1].split('_')[0] for f in glob.glob(os.path.join(input_dir, 'three_band', '*.tif'))))

train_image_ids = ['6040_2_2', '6120_2_2', '6120_2_0', '6090_2_0', '6040_1_3', '6040_1_0', '6100_1_3', '6010_4_2', '6110_4_0', '6140_3_1', '6110_1_2', '6100_2_3', '6150_2_3', '6160_2_1', '6140_1_2', '6110_3_1', '6010_4_4', '6170_2_4', '6170_4_1', '6170_0_4', '6060_2_3', '6070_2_3', '6010_1_2', '6040_4_4', '6100_2_2']

mask_size = 1024
image_size = 1024
border_size = 0
