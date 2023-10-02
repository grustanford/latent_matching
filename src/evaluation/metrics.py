"""scoring metrics"""
import time
import numpy as np
from scipy.stats import zscore
from pysaliency.roc import general_roc

# parameters
params = {
    # index parameters
    'indexing_by_eval': False,

    # scan order parameters 
    'scan_orders': [0,1,2,3,4], # None means all scan orders except 0th fixation

    # image parameters
    'downsample_factor': 105,
    'n_w': 16,  # map width 
    'n_h': 10,  # map height

    # print parameters
    'verbose': True,
    'print_after_n': 100,
}

    
def get_positions(scanpaths, idx_img, idx_tgt, idx_eval, idx_include=None, negative=True, params=params, **kwargs):
    """
    Obtain x,y positions of the fixations.
    x,y coordinates correspond to the axes along with the width and height of images, respectively.

    inputs
        scanpaths [n_scanpaths]: list of scanpath dictionaries.
        idx_img [n_scanpaths]: image indices of scanpaths.
        idx_tgt [n_scanpaths]: target indices of scanpaths.
        idx_eval [n_tasks,2]: task (image-target pair) index to be evaluated.
        idx_include [n_scanpaths]: list of boolean indices to include in the scoring. None means all inclusion.
        params: dictionary of related parameters

    return
        xy_pos [n_tasks]: precomputed list of fixations (x,y) corresponding to the positive distribution
        xy_neg [n_tasks]: precomputed list of fixations (x,y) corresponding to the negative distribution (optional)
    """

    s_time   = time.time()
    n_tasks  = len(idx_eval)
    n_orders = 1 if params['scan_orders'] is None else len(params['scan_orders'])

    # obtain x,y positions [with filters]
    if params['scan_orders'] is None:
        if idx_include is None:
            idx_include = [ np.ones(s['length'], dtype=bool) for s in scanpaths ]

        xs = [ [ss for ss,vv in zip(s['X'][1:], v[1:]) if vv] for s,v in zip(scanpaths, idx_include) ]
        ys = [ [ss for ss,vv in zip(s['Y'][1:], v[1:]) if vv] for s,v in zip(scanpaths, idx_include) ]
        
    else:
        if idx_include is None:
            idx_include = np.ones((len(scanpaths),n_orders), dtype=bool)

        xs = [ np.array([s['X'][v_o] if (len(s['X']) > v_o) & v else np.nan for s,v in zip(scanpaths, idx_include[:,i_o])]) for i_o, v_o in enumerate(params['scan_orders']) ]
        ys = [ np.array([s['Y'][v_o] if (len(s['Y']) > v_o) & v else np.nan for s,v in zip(scanpaths, idx_include[:,i_o])]) for i_o, v_o in enumerate(params['scan_orders']) ]
            
    #
    if negative:
        xy_pos, xy_neg = [], []
    else:
        xy_pos = []

    for i_eval, v_eval in enumerate(idx_eval):
        
        # select the task (img - target pair) to be evaluated
        img, tgt = v_eval
        idxp = (idx_img==img) & (idx_tgt==tgt) # index for positives

        if negative:
            idxn = (idx_img!=img) & (idx_tgt!=tgt) # index for negatives
            _xy_pos, _xy_neg = [], []
        else:
            _xy_pos = []

        #
        for i_o in range(n_orders):

            if params['scan_orders'] is None:
                xp = np.concatenate( [s if v else [] for s,v in zip(xs,idxp)] )
                yp = np.concatenate( [s if v else [] for s,v in zip(ys,idxp)] )
                if negative:
                    xn = np.concatenate( [s if v else [] for s,v in zip(xs,idxn)] )
                    yn = np.concatenate( [s if v else [] for s,v in zip(ys,idxn)] )

            else:
                xp, yp = xs[i_o][idxp], ys[i_o][idxp]
                if negative:
                    xn, yn = xs[i_o][idxn], ys[i_o][idxn]
            
            # obtain positive fixation positions
            nnan      = (~np.isnan(xp)) & (~np.isnan(yp))
            xp        = (xp[nnan] // params['downsample_factor']).astype(int) 
            yp        = (yp[nnan] // params['downsample_factor']).astype(int) 
            nnan      = (xp >= 0) & (xp < params['n_w']) & (yp >= 0) & (yp < params['n_h'])
            xp, yp    = xp[nnan], yp[nnan]
            _xy_pos.append(np.array([xp, yp], dtype=int))

            # obtain negative fixation positions
            if negative:
                nnan      = (~np.isnan(xn)) & (~np.isnan(yn))
                xn        = (xn[nnan] // params['downsample_factor']).astype(int) 
                yn        = (yn[nnan] // params['downsample_factor']).astype(int) 
                nnan      = (xn >= 0) & (xn < params['n_w']) & (yn >= 0) & (yn < params['n_h'])
                xn, yn    = xn[nnan], yn[nnan]
                _xy_neg.append(np.array([xn, yn], dtype=int))

        xy_pos.append(_xy_pos)
        if negative:
            xy_neg.append(_xy_neg)
        
        if params['verbose'] & ((i_eval+1) % params['print_after_n'] == 0):
            print( 'completed: %d/%d' % (i_eval+1, n_tasks) )
    
    e_time = time.time()

    if params['verbose']: 
        print('total elapsed time: %.2f sec' % (e_time - s_time))

    if negative:
        return {
            'xy_pos': xy_pos,
            'xy_neg': xy_neg
        }
    else: 
        return {
            'xy_pos': xy_pos
        }


def nss(maps, fix_pos=None, idx_eval=None, idx_img=None, idx_tgt=None, idx_include=None, scanpaths=None, params=params, **kwargs):
    """
    Normalized scanpath saliency (NSS) score.

    inputs
        maps [n_imgs, n_h, n_w] or [n_imgs, n_layers, n_h, n_w, n_tgts]: array of predictive maps.
        fix_pos [n_tasks]: precomputed list of fixations (x,y) corresponding to the positive distribution
        idx_eval [n_tasks,2]: task (image-target pair) index to be evaluated.
        idx_img [n_scanpaths]: image indices of scanpaths.
        idx_tgt [n_scanpaths]: target indices of scanpaths.
        scanpaths [n_scanpaths]: list of scanpath dictionaries.
        idx_include [n_scanpaths]: list of boolean indices to include in the scoring. None means all inclusion.

    return
        scores [n_tasks] or [n_scan_orders, n_tasks]
    """

    s_time   = time.time()
    n_tasks  = len(idx_eval)
    n_orders = 1 if params['scan_orders'] is None else len(params['scan_orders'])

    if len(maps.shape) == 3:
        maps = maps[:,np.newaxis,:,:,np.newaxis]

    _, n_layers, n_h, n_w, n_tgts = maps.shape

    #
    if fix_pos is None:
        print( "precomputed fixations not found. obtaining fixations..." )
        fix_pos = get_positions(scanpaths=scanpaths, 
                                idx_img=idx_img, 
                                idx_tgt=idx_tgt, 
                                idx_eval=idx_eval, 
                                idx_include=idx_include, 
                                negative=False, params=params).values()

    # z-scoring
    maps = zscore(maps.reshape((-1,n_layers,n_h*n_w,n_tgts)), axis=2, nan_policy='omit').reshape((-1,n_layers,n_h,n_w,n_tgts))

    # 
    scores = np.nan*np.zeros((n_orders,n_layers,n_tasks))

    for i_eval, v_eval in enumerate(idx_eval):

        # select the task (img - target pair) to be evaluated
        img, tgt = v_eval
        if n_tgts == 1: tgt = 0

        # taking the means of positives
        for i_o in range(n_orders):

            # obtain x,y positions
            xp, yp = fix_pos[i_eval][i_o]

            positives = maps[i_eval,:,:,:,tgt] if params['indexing_by_eval'] else maps[img,:,:,:,tgt]
            positives = positives[:,yp,xp]

            # computation of nss
            scores[i_o,:,i_eval] = np.mean(positives,axis=-1)
        
        if params['verbose'] & ((i_eval+1) % params['print_after_n'] == 0):
            print( 'completed: %d/%d' % (i_eval+1, n_tasks) )

    scores = scores.squeeze()
    
    e_time = time.time()
    if params['verbose']: 
        print('total elapsed time: %.2f sec' % (e_time - s_time))

    return scores


def auc(maps, fix_pos=None, idx_eval=None, idx_img=None, idx_tgt=None, idx_include=None, scanpaths=None, params=params, **kwargs):
    """
    Area under the ROC curve (AUC) score, with the negative distribution generated from uniform sampling of the task predictive map responses.

    inputs
        maps [n_imgs, n_h, n_w] or [n_imgs, n_layers, n_h, n_w, n_tgts]: array of predictive maps.
        fix_pos [n_tasks]: precomputed list of fixations (x,y) corresponding to the positive distribution
        idx_eval [n_tasks,2]: task (image-target pair) index to be evaluated.
        idx_img [n_scanpaths]: image indices of scanpaths.
        idx_tgt [n_scanpaths]: target indices of scanpaths.
        idx_include [n_scanpaths]: list of boolean indices to include in the scoring. None means all inclusion.
        scanpaths [n_scanpaths]: list of scanpath dictionaries.

    return
        scores [n_tasks] or [n_scan_orders, n_tasks]
    """

    s_time   = time.time()
    n_tasks  = len(idx_eval)
    n_orders = 1 if params['scan_orders'] is None else len(params['scan_orders'])

    # 
    if len(maps.shape) == 3:
        maps = maps[:,np.newaxis,:,:,np.newaxis]

    _, n_layers, _, _, n_tgts = maps.shape

    if fix_pos is None:
        print( "precomputed fixations not found. obtaining fixations..." )
        fix_pos = get_positions(scanpaths=scanpaths, 
                                idx_img=idx_img, 
                                idx_tgt=idx_tgt, 
                                idx_eval=idx_eval, 
                                idx_include=idx_include, 
                                negative=False, params=params).values()
                
    # 
    scores = np.nan*np.zeros((n_orders,n_layers,n_tasks))

    for i_eval, v_eval in enumerate(idx_eval):
        
        # select the task (img - target pair) to be evaluated
        img, tgt = v_eval
        if n_tgts == 1: tgt = 0

        # generation of negatives (uniform)
        negatives = maps[i_eval,:,:,:,tgt] if params['indexing_by_eval'] else maps[img,:,:,:,tgt]
        negatives = negatives.reshape((n_layers,-1))

        # generation of positives
        for i_o in range(n_orders):

            # obtain x,y positions
            xp, yp = fix_pos[i_eval][i_o]

            positives = maps[i_eval,:,:,:,tgt] if params['indexing_by_eval'] else maps[img,:,:,:,tgt]
            positives = positives[:,yp,xp]

            # computation of auc
            for i_layer in range(n_layers):
                auc, _, _ = general_roc(positives[i_layer].astype(float), negatives[i_layer].astype(float), judd=0)
                scores[i_o,i_layer,i_eval] = auc

        if params['verbose'] & ((i_eval+1) % params['print_after_n'] == 0):
            print( 'completed: %d/%d' % (i_eval+1, n_tasks) )
    
    scores = scores.squeeze()

    e_time = time.time()
    if params['verbose']: 
        print('total elapsed time: %.2f sec' % (e_time - s_time))

    return scores


def sauc(maps, fix_pos=None, fix_neg=None, idx_eval=None, idx_img=None, idx_tgt=None, idx_include=None, scanpaths=None, params=params, **kwargs):
    """
    Suffled AUC score, with the negative distribution generated from sampling of fixations from the other tasks. 

    inputs
        maps [n_imgs, n_h, n_w] or [n_imgs, n_layers, n_h, n_w, n_tgts]: array of predictive maps.
        fix_pos [n_tasks]: precomputed list of fixations (x,y) corresponding to the positive distribution
        fix_pos [n_tasks]: precomputed list of fixations (x,y) corresponding to the negative distribution
        idx_img [n_scanpaths]: image indices of scanpaths.
        idx_tgt [n_scanpaths]: target indices of scanpaths.
        idx_eval [n_tasks,2]: task (image-target pair) index to be evaluated.
        idx_include [n_scanpaths]: list of boolean indices to include in the scoring. None means all inclusion.

    return
        scores [n_tasks] or [n_scan_orders, n_tasks]
    """

    s_time   = time.time()
    n_tasks  = len(idx_eval)
    n_orders = 1 if params['scan_orders'] is None else len(params['scan_orders'])

    # 
    if len(maps.shape) == 3:
        maps = maps[:,np.newaxis,:,:,np.newaxis]

    _, n_layers, _, _, n_tgts = maps.shape

    # 
    if fix_pos is None:
        print( "precomputed fixations not found. obtaining fixations..." )
        fix_pos, fix_neg = get_positions(scanpaths=scanpaths, 
                                         idx_img=idx_img, 
                                         idx_tgt=idx_tgt, 
                                         idx_eval=idx_eval, 
                                         idx_include=idx_include, 
                                         negative=True, params=params).values()

    # 
    scores = np.nan*np.zeros((n_orders,n_layers,n_tasks))

    for i_eval, v_eval in enumerate(idx_eval):
        
        # select the task (img - target pair) to be evaluated
        img, tgt = v_eval
        if n_tgts == 1: tgt = 0

        #
        for i_o in range(n_orders):

            # obtain x,y positions
            xp, yp = fix_pos[i_eval][i_o]
            xn, yn = fix_neg[i_eval][i_o]

            negatives = maps[i_eval,:,:,:,tgt] if params['indexing_by_eval'] else maps[img,:,:,:,tgt]
            negatives = negatives[:,yn,xn]
            positives = maps[i_eval,:,:,:,tgt] if params['indexing_by_eval'] else maps[img,:,:,:,tgt]
            positives = positives[:,yp,xp]

            # computation of auc
            for i_layer in range(n_layers):
                auc, _, _ = general_roc(positives[i_layer].astype(float), negatives[i_layer].astype(float), judd=0)
                scores[i_o,i_layer,i_eval] = auc

        if params['verbose'] & ((i_eval+1) % params['print_after_n'] == 0):
            print( 'completed: %d/%d' % (i_eval+1, n_tasks) )
        
    scores = scores.squeeze()

    e_time = time.time()
    if params['verbose']: 
        print('total elapsed time: %.2f sec' % (e_time - s_time))

    return scores
