import pandas as pd
import os

from .meta import input_dir

train_wkt = pd.read_csv(os.path.join(input_dir, 'train_wkt_v4.csv'), names=['image_id', 'cls', 'multi_poly_wkt'], skiprows=1)
grid_sizes = pd.read_csv(os.path.join(input_dir, 'grid_sizes.csv'), names=['image_id', 'xmax', 'ymin'], skiprows=1, index_col='image_id')

sample_submission = pd.read_csv(os.path.join(input_dir, 'sample_submission.csv'))
