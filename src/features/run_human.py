"""make inter-subject consistency density maps with Gaussian smoothing
    saving the output files is NOT recommended. it will take a lot of space.
    rather, this script can be used on the fly along with the evaluation script.
"""
# TODO: normalize the density maps whose sum should be 1
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import pickle
import argparse
import multiprocessing
import numpy as np
from pathlib import Path
from functools import partial
from scipy.ndimage import gaussian_filter

sys.path.append( str(Path(os.path.dirname(__file__)).parent.absolute()) )
from data import load
from evaluation import metrics

# argparser
parser = argparse.ArgumentParser(description='Build density maps for human fixations')
parser.add_argument('--data_split',  type=str, default='trainval', help='data split')
parser.add_argument('--path',        type=str, default='data/processed', help='path to the processed data')
parser.add_argument('--path_out',    type=str, default='data/processed', help='path to the output data')
parser.add_argument('--path_data',   type=str, default='data/external/coco_search18', help='path to the external data')
parser.add_argument('--save_name',   type=str, default='human', help='save name')
parser.add_argument('--kernel_size', type=float, default=30., help='kernel size for Gaussian smoothing')
parser.add_argument('--scan_orders',  nargs='+', type=int, default=None, help='scan orders to consider. None for all except initial')
parser.add_argument('--downsample_factor', type=int, default=1, help='downsample factor')
parser.add_argument('--collapse_subjects', type=bool, default=False, help='collapse subjects')
parser.add_argument('--normalize_between', type=bool, default=False, help='normalize between subjects to sum to 1')
parser.add_argument('--normalize_within',  type=bool, default=False, help='normalize within subjects to sum to 1')
parser.add_argument('--n_jobs', type=int, default=-1, help='multiprocessing jobs. 0/- means all cpus but the specified #') 

args = parser.parse_args()
data_split        = args.data_split
path_prcd         = args.path
path_out          = args.path_out
path_data         = args.path_data
save_name         = args.save_name
kernel_size       = args.kernel_size
scan_orders       = args.scan_orders
downsample_factor = args.downsample_factor
collapse_subjects = args.collapse_subjects
normalize_between = args.normalize_between
normalize_within  = args.normalize_within
n_jobs            = args.n_jobs

# parameters
# params = {
#     # image parameters
#     'img_w': 1680,
#     'img_h': 1050,
# }

# load data
data_dict, sbj_dict = load.load_indices(
    path_prcd     = path_prcd, 
    path_scanpath = path_data, 
    data_split    = data_split,
    merge         = True
)
LIST_SBJ = np.unique(data_dict['idx_sbj'])

# load image indices
with open(Path(path_prcd)/'images'/f'indices_{data_split}.pkl', 'rb') as f:
    images_idx = pickle.load(f)

# get positions [only positives]
metrics.params['n_w'] = 1680
metrics.params['n_h'] = 1050
metrics.params['scan_orders'] = scan_orders
metrics.params['downsample_factor'] = downsample_factor
metrics.params['verbose'] = False

# subject-wise fixation positions
path_output = Path(path_out) / 'maps' / save_name
path_output.mkdir(parents=True, exist_ok=True)
path_positions = path_output / f'fixations_{data_split}.pkl'

if not path_positions.exists():
    print('####### Fixation positions not found. Generating... #######')
    print('####### This may take a while... #######')

    positions = []
    for v_sub in LIST_SBJ:
        if metrics.params['scan_orders'] is None:
            sub_include = [ np.repeat(data_dict['idx_sbj'][i]==v_sub, s['length']) for i,s in enumerate(data_dict['scanpaths']) ]
        else:
            sub_include = np.array([ np.repeat(data_dict['idx_sbj'][i]==v_sub, len(metrics.params['scan_orders'])) for i,s in enumerate(data_dict['scanpaths']) ])
        sub_positions = metrics.get_positions(**data_dict, idx_include=sub_include, negative=False)
        positions.append( sub_positions )

    with open(path_positions, 'wb') as f:
        pickle.dump(positions, f)

with open(path_positions, 'rb') as f:
    positions = pickle.load(f)


# loop
path_output = Path(path_out) / 'maps' / save_name / data_split
path_output.mkdir(parents=True, exist_ok=True)

def loop(i_img, v_img):
    maps = {}
    idx_idx = data_dict['idx_eval'][:, 0] == i_img
    for i_eval, (_, tgt) in enumerate(data_dict['idx_eval'][idx_idx]):
        maps[tgt] = {}
        for i_sub, v_sub in enumerate(LIST_SBJ):

            if scan_orders is None:
                v_map = np.zeros([metrics.params['n_h'], metrics.params['n_w']])
                _x, _y = positions[i_sub]['xy_pos'][i_eval][0]
                if len(_x) == 0: v_map = np.nan * v_map
                for _xx, _yy in zip(_x, _y):
                    v_map[_yy, _xx] += 1.
                v_map = gaussian_filter(v_map, kernel_size)

            else:
                v_map = np.zeros([metrics.params['n_h'], metrics.params['n_w'], len(scan_orders)])
                for i_o, v_o in enumerate(scan_orders):
                    _x, _y = positions[i_sub]['xy_pos'][i_eval][i_o]
                    if len(_x) == 0: v_map[:, :, i_o] = np.nan * v_map[:, :, i_o]
                    for _xx, _yy in zip(_x, _y):
                        v_map[_yy, _xx, i_o] += 1.
                    v_map[:,:,i_o] = gaussian_filter(v_map[:, :, i_o], kernel_size)

            maps[tgt][v_sub] = v_map

    with open(path_output / (v_img.split('.')[0] + 'pkl'), 'wb') as f:
        pickle.dump(maps, f)

# multiprocessing
if n_jobs <= 0:
    n_jobs = multiprocessing.cpu_count() + n_jobs
print( 'Using', n_jobs, 'jobs...' )

with multiprocessing.Pool() as pool:
    pool.starmap(partial(loop), enumerate(images_idx))