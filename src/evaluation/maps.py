"""evalaute maps with given metrics"""
# TODO: complete this script
import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import zscore

sys.path.append( str(Path(os.path.dirname(__file__)).parent.absolute()) )
from data import load
from evaluation import metrics

LIST_SBJ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

parser = argparse.ArgumentParser(description='Evaluate maps')
parser.add_argument('--data_split',   type=str, default='trainval', help='data split')
parser.add_argument('--path_data',    type=str, default='data/external/coco_search18', help='path to the external data')
parser.add_argument('--path',         type=str, default='data/processed', help='path to the processed data')
parser.add_argument('--metric',       type=str, default='sauc', help='metric to evaluate')
parser.add_argument('--prefix',       type=str, default='agg', help='prefix for the output file')
parser.add_argument('--include',      type=str, default='all', help='inclusion criteria')
parser.add_argument('--map',          nargs='+',  type=str, default='smm', help='maps to evaluate')
parser.add_argument('--scan_orders',  nargs='+', type=int, default=None, help='scan orders to consider. None for all except initial')
parser.add_argument('--mixture',      type=bool, default=False, help='whether to consider complementary maps')
parser.add_argument('--map_mixture',  nargs='+', type=str, default=None, help='maps to evaluate for mixture')

args = parser.parse_args()
data_split  = args.data_split
path_data   = args.path_data
path_prcd   = args.path
metric      = args.metric
prefix      = args.prefix
include     = args.include
name_maps   = args.map
scan_orders = args.scan_orders

mixture           = args.mixture
name_maps_mixture = args.map_mixture
if mixture:
    """complementary maps parameters"""
    MIX_WEIGHTS = np.linspace(0,1,num=101) # weights for SMM
    MIX_LAYER   = 23 # penultimate layer index for SMM
    
# TODO: exception handling
if include == 'all':
    idx_include = None


print('####### Evaluation #######')
print(name_maps)

# load data
print('####### Loading data #######')
tp_dict, ta_dict, sbj_dict = load.load_indices(
    path_index=path_prcd, path_scanpath=path_data, data_split=data_split
)
metrics.params['scan_orders'] = scan_orders


# load metric functions
if metric == 'nss':
    metric_func = metrics.nss
elif metric == 'auc':
    metric_func = metrics.auc
elif metric == 'sauc':
    metric_func = metrics.sauc
else:
    raise ValueError('metric not supported')


# load / generate fixation positions
_path_positions = Path(path_prcd)/ f'scores/{prefix}_fixations_{data_split}.pkl'
if not _path_positions.exists():
    print('####### Fixation positions not found. Generating... #######')
    fixations = {}
    fixations['tp'] = metrics.get_positions(**tp_dict, idx_include=idx_include)
    fixations['ta'] = metrics.get_positions(**ta_dict, idx_include=idx_include)
    with open(_path_positions, 'wb') as f:
        pickle.dump(fixations, f)

with open(_path_positions, 'rb') as f:
    fixations = pickle.load(f)


# evaluate maps
# TODO : cannot load maps as matrix. Need to load each map separately.
if not mixture:
    print('####### Evaluating maps #######')
    for nmap in name_maps:
        print('#######', nmap, '#######')

        # load map
        # TODO: humans do not correspond to this format
        with open( Path(path_prcd)/f'maps/{nmap}/{data_split}.pkl', 'rb' ) as f:
            maps = pickle.load(f)
        maps  = np.stack([v for _,v in maps.items()], axis=0)

        # evaluate
        _path_score = Path(path_prcd)/f'scores/{nmap}'
        _path_score.mkdir(parents=True, exist_ok=True)

        score = {}
        if nmap == 'human':
            metrics.params['indexing_by_eval'] = True
            for k in ['tp', 'ta']:
                score_s = []  
                for i_sbj, _ in enumerate(LIST_SBJ):
                    score_s.append( metric_func( **sbj_dict[k][i_sbj], maps=maps[k][i_sbj] ) )
                score[k] = np.nanmean(np.stack(score_s,axis=-1),axis=-1)
            metrics.params['indexing_by_eval'] = False
        else:
            score['tp'] = metric_func(**tp_dict, **fixations['tp'], maps=maps)
            score['ta'] = metric_func(**ta_dict, **fixations['ta'], maps=maps)

        with open( _path_score/f'{prefix}_{metric}_{data_split}.pkl', 'wb' ) as f:
            pickle.dump(score, f)

else:
    print('####### Evaluating complementary maps #######')

    metrics.params['verbose'] = False
    def zscore_maps(maps):
        _shape   = maps.shape
        _reshape = [_shape[0], _shape[1]*_shape[2], -1]
        return zscore(maps.reshape(_reshape), axis=1, nan_policy='omit').reshape(_shape)

    for nmap1, nmap2 in zip(name_maps, name_maps_mixture):
        print('####### Preparing complementary maps:', nmap1, nmap2, '#######')
        # map1 (smm)
        with open( Path(path_prcd)/f'maps/{nmap1}/{data_split}.pkl', 'rb' ) as f:
            maps1 = pickle.load(f)
        maps1 = np.stack([v for k,v in maps1.items()], axis=0)
        maps1 = maps1[:,MIX_LAYER]
        maps1 = zscore_maps(maps1)

        # map2 (saliency maps)
        with open( Path(path_prcd)/f'maps/{nmap2}/{data_split}.pkl', 'rb' ) as f:
            maps2 = pickle.load(f)
        maps2 = np.stack([v for k,v in maps2.items()], axis=0)
        maps2 = zscore_maps(maps2)

        # preparing maps
        mapsw = []
        for w in MIX_WEIGHTS:
            mapsw.append( w*maps1 + (1.-w)*maps2 )
        mapsw = np.stack(mapsw)[:,:,np.newaxis]

        print('####### Preparing maps completed. Now computing scores... #######')
        score = { 'tp':[], 'ta':[] }
        for i_map, v_map in enumerate(mapsw):
            score['tp'].append( metric_func(maps=v_map, **tp_dict, **fixations['tp']) )
            score['ta'].append( metric_func(maps=v_map, **ta_dict, **fixations['ta']) )
            print(f"{i_map:03}th: completed")
        score['tp'] = np.stack(score['tp'])
        score['ta'] = np.stack(score['ta'])

        _path_score = Path(path_prcd)/f'scores/complementary'
        _path_score.mkdir(parents=True, exist_ok=True)
        with open( _path_score/f'{prefix}_{nmap1}_{nmap2}_{metric}_{data_split}.pkl', 'wb' ) as f:
            pickle.dump(score, f)






# inclusion criteria
