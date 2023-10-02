"""make DeepGaze-IIE (Linardos & Kümmerer et al., 2021) predictive maps"""
import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import zoom
from scipy.special import logsumexp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

sys.path.append( str(Path(os.path.dirname(__file__)).parent.absolute()) )
from data import load

parser = argparse.ArgumentParser(description='Build maps for DeepGaze-IIE model')
parser.add_argument('--data_split',  type=str, default='trainval', help='data split')
parser.add_argument('--path',        type=str, default='data/processed', help='path to the processed data')
parser.add_argument('--path_data',   type=str, default='data/external/coco_search18', help='path to the external data')
parser.add_argument('--save_name',   type=str, default='deepgaze', help='save name')
parser.add_argument('--model_dir',   type=str, default='models/DeepGaze', help='model directory')
parser.add_argument('--cache_dir',   type=str, default='models/checkpoints', help='cache directory')
parser.add_argument('--device',      type=str, default=None, help='device')
parser.add_argument('--batch_size',  type=int, default=16, help='batch size')
parser.add_argument('--center_bias', type=bool, default=True, help='whether to consider center bias (MIT1003)')

args = parser.parse_args()
data_split  = args.data_split
path_prcd   = args.path
path_data   = args.path_data
save_name   = args.save_name
model_dir   = args.model_dir
cache_dir   = args.cache_dir
batch_size  = args.batch_size
center_bias = args.center_bias

device = args.device
if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
params = {
    # image parameters
    'img_w': 1680,
    'img_h': 1050,
}

# load model
sys.path.append(model_dir)
import deepgaze_pytorch

torch.hub.set_dir( Path(cache_dir)/'torch/hub' )
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
model.eval()
print('DeepGaze-IIE loaded, using device:', device)


# data loader
files = load.load_images(path_data, data_split=data_split, save_images=False)

transform = transforms.Compose([
    transforms.PILToTensor(),
])

class ImageDataset(Dataset):
    """Image dataset"""
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        """load images"""
        path = self.path[idx]
        img  = Image.open(path).convert('RGB')
        img  = self.transform(img).to(device)
        return img, idx

dataloader = DataLoader(
    ImageDataset(files, transform), 
    batch_size=batch_size
)


# load center bias
if center_bias:
    cbias = np.load('data/external/MIT1003/centerbias_mit1003.npy')
    cbias = zoom(cbias, (params['img_h']/cbias.shape[0], params['img_w']/cbias.shape[1]), order=0, mode='nearest')
else:
    cbias = np.zeros((params['img_h'], params['img_w']))

cbias -= logsumexp(cbias)
cbias  = torch.tensor(cbias)[None].to(device)


# loop 
path_output = Path(path_prcd) / 'maps' / save_name / data_split
path_output.mkdir(parents=True, exist_ok=True)

for i, (img, idx) in enumerate(dataloader):
    with torch.no_grad():
        log_density = model(img, cbias)
    
    # save
    data_files  = [Path(f).name for f in files[idx]]
    log_density = log_density.detach().cpu().numpy()

    for fn,ld in zip( data_files,log_density ):
        with open( path_output / (fn.split('.')[0] + '.pkl'), 'wb' ) as f:
            pickle.dump( ld[0], f )
            
    print('complete: ', i+1, '/', len(dataloader), 'batches')