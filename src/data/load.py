import glob
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

# 
def load_scanpaths(path, data_split='trainval'):
    """
    load scanpaths from the path.

    inputs
        path: path to coco-search18 dataset
        data_split: 'trainval', 'test'

    return
        scanpaths_tp [list] : list of TP scanpaths
        scanpaths_ta [list] : list of TA scanpaths
    """

    dir_tp = Path(path)/'tp/scanpaths'
    dir_ta = Path(path)/'ta/scanpaths'
    
    if data_split == 'trainval':

        with open(dir_tp/'coco_search18_fixations_TP_train_split1.json', 'r') as f:
            scanpaths_tp1 = json.load(f)
        with open(dir_tp/'coco_search18_fixations_TP_validation_split1.json', 'r') as f:
            scanpaths_tp2 = json.load(f)
        scanpaths_tp = scanpaths_tp1 + scanpaths_tp2
        del scanpaths_tp1, scanpaths_tp2

        with open(dir_ta/'coco_search18_fixations_TA_trainval.json', 'r') as f:
            scanpaths_ta = json.load(f)

    elif data_split == 'test':
        with open(dir_tp/'coco_search18_fixations_TP_test.json', 'r') as f:
            scanpaths_tp = json.load(f)

        with open(dir_ta/'coco_search18_fixations_TA_test.json', 'r') as f:
            scanpaths_ta = json.load(f)

    return scanpaths_tp, scanpaths_ta


def load_images(path, path_outputs=None, data_split='trainval'):
    """
    load images from the path.

    inputs
        path: path to coco-search18 dataset
        path_outputs: path to save the images. None if not saving.
        data_split: 'trainval', 'test'

    return
        images [dict] : dictionary of PIL images
        images_idx [list] : list of image names
    """

    # load scanpaths
    scanpaths_tp, scanpaths_ta = load_scanpaths(path, data_split=data_split)

    # list all the image names
    dir_tp = Path(path)/'tp/images'
    dir_ta = Path(path)/'ta/images'
    files_tp = glob.glob(dir_tp/'**/*.jpg', recursive=True)
    files_ta = glob.glob(dir_ta/'**/*.jpg', recursive=True)
    files = np.array(files_tp + files_ta)

    # 
    names = np.unique([s['name'] for s in scanpaths_ta+scanpaths_tp])
    images = {}
    for v_jpg in names:
        path_images = [f for f in files if v_jpg in f]
        images[ v_jpg ] = Image.open(path_images[0])
    images_idx = np.array(list(images.keys()))

    # saving images
    if path_outputs is not None: 

        path_output_images = Path(path_outputs)/'images'
        path_output_images.mkdir(parents=True, exist_ok=True)
        
        with open(path_output_images/f'images/images_{data_split}.pkl', 'wb') as f:
            pickle.dump(images, f)
        with open(path_output_images/f'images/indices_{data_split}.pkl', 'wb') as f:
            pickle.dump(images_idx, f)

    return images, images_idx


# 
def load_indices(scanpaths, images_idx=None, path_index=None, data_split='trainval'):
    """
    load indices from the path.
    inputs
        scanpaths [tp,ta] : list of scanpaths
        images_idx [list] : list of image indices
        path_index: path to image indices
        data_split: 'trainval', 'test'
    
    return
        tp_dict [dict] : dictionary of TP indices, required for score evaluation
        ta_dict [dict] : dictionary of TA indices, required for score evaluation
        sbj_dict [dict] : dictionary of subject indices, required for score evaluation
    """

    if images_idx is None and path_index is not None:
        with open(Path(path_index)/'images'/f'indices_{data_split}.pkl', 'rb') as f:
            images_idx = pickle.load(f)

    # load scanpaths
    scanpaths_tp, scanpaths_ta = scanpaths

    #
    tp_img = [s['name'] for s in scanpaths_tp]
    tp_tgt = [s['task'] for s in scanpaths_tp]
    ta_img = [s['name'] for s in scanpaths_ta]
    ta_tgt = [s['task'] for s in scanpaths_ta]

    list_tgt = np.unique([sc['task'] for sc in scanpaths_tp+scanpaths_ta])
    tp_img_idx = np.array([np.where(i==images_idx)[0][0] for i in tp_img])
    ta_img_idx = np.array([np.where(i==images_idx)[0][0] for i in ta_img])
    tp_tgt_idx = np.array([np.where(t==list_tgt)[0][0] for t in tp_tgt])
    ta_tgt_idx = np.array([np.where(t==list_tgt)[0][0] for t in ta_tgt])

    # make index for evaluation
    idx_eval_tp = []
    img_list = np.unique(tp_img_idx)
    for v_img in img_list:    
        # list the corresponding tasks (indexes)
        tgt_list = np.unique( tp_tgt_idx[tp_img_idx==v_img] )
        
        # loop through the corresponding tasks (indexes)
        for v_tgt in tgt_list:
            idx_eval_tp.append([v_img, v_tgt])
    idx_eval_tp = np.array(idx_eval_tp)

    # 
    idx_eval_ta = []
    img_list = np.unique(ta_img_idx)
    for v_img in img_list:
        # list the corresponding tasks (indexes)
        tgt_list = np.unique( ta_tgt_idx[ta_img_idx==v_img] )
        
        # loop through the corresponding tasks (indexes)
        for v_tgt in tgt_list:
            idx_eval_ta.append([v_img, v_tgt])
    idx_eval_ta = np.array(idx_eval_ta)

    # 
    tp_dict = {
        'idx_img': tp_img_idx,
        'idx_tgt': tp_tgt_idx,
        'scanpaths': scanpaths_tp, 
        'idx_eval': idx_eval_tp,
    }
    ta_dict = {
        'idx_img': ta_img_idx,
        'idx_tgt': ta_tgt_idx,
        'scanpaths': scanpaths_ta, 
        'idx_eval': idx_eval_ta,
    }

    # 
    sbj_tp = np.array([s['subject'] for s in scanpaths_tp])
    sbj_ta = np.array([s['subject'] for s in scanpaths_ta])
    list_sbj = np.unique(sbj_tp)

    tp_dict_sbj = []
    ta_dict_sbj = []
    for v_sbj in list_sbj:
        idx_sbj_tp = (sbj_tp==v_sbj)
        tp_dict_sbj.append({
            'idx_img'  : [v for v,t in zip(tp_img_idx,  idx_sbj_tp) if t],
            'idx_tgt'  : [v for v,t in zip(tp_tgt_idx,  idx_sbj_tp) if t],
            'scanpaths': [v for v,t in zip(scanpaths_tp,idx_sbj_tp) if t], 
            'idx_eval' : idx_eval_tp,
        })
        
        idx_sbj_ta = (sbj_ta==v_sbj)
        ta_dict_sbj.append({
            'idx_img'  : [v for v,t in zip(ta_img_idx,  idx_sbj_ta) if t],
            'idx_tgt'  : [v for v,t in zip(ta_tgt_idx,  idx_sbj_ta) if t],
            'scanpaths': [v for v,t in zip(scanpaths_ta,idx_sbj_ta) if t], 
            'idx_eval' : idx_eval_ta,
        })

    sbj_dict = {
        'tp': tp_dict_sbj,
        'ta': ta_dict_sbj
    }

    return tp_dict, ta_dict, sbj_dict