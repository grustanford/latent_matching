"""make semantic similarity maps"""
import os
import sys
import pickle
import argparse
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append( str(Path(os.path.dirname(__file__)).parent.absolute()) )
from data import load

parser = argparse.ArgumentParser(description='Build maps for semantic searcher model')
parser.add_argument('--data_split',  type=str, default='trainval', help='data split')
parser.add_argument('--path',        type=str, default='data/processed', help='path to the processed data')
parser.add_argument('--path_data',   type=str, default='data/external/coco_search18', help='path to the external data')
parser.add_argument('--save_name',   type=str, default='smm', help='save name')
parser.add_argument('--model_name',  type=str, default='openai/clip-vit-large-patch14', help='model name')
parser.add_argument('--custom_name', type=str, default=None, help='custom model name')
parser.add_argument('--cache_dir',   type=str, default='models/checkpoints', help='cache directory')
parser.add_argument('--device',      type=str, default=None, help='device')
parser.add_argument('--batch_size',  type=int, default=3, help='batch size')
parser.add_argument('--high_res',    type=bool, default=True, help='whether to return high resolution maps')

args = parser.parse_args()
data_split  = args.data_split
path_prcd   = args.path
path_data   = args.path_data
model_name  = args.model_name
custom_name = args.custom_name
save_name   = args.save_name
cache_dir   = args.cache_dir
high_res    = args.high_res
batch_size  = args.batch_size

if custom_name is None:
    custom_name = model_name
else:
    custom_name = str( Path(cache_dir)/'huggingface/hub/'/custom_name )

device = args.device
if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
params = {
    # image parameters
    'img_w': 1680,
    'img_h': 1050,

    # margin parameters
    'margin': 14,
    'stride': 1,
}

# load model
os.environ['TRANSFORMERS_CACHE'] = str( Path(cache_dir)/'huggingface/hub/' )
from smm import SemanticMatchingModel
from transformers import CLIPProcessor

model = SemanticMatchingModel.from_pretrained(custom_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()
with torch.no_grad(): model.set_prototype(processor)  # set prototype target vector
print('Semantic matching model loaded, using device:', device)


# high resolution maps
if high_res:
    SIZE_IMAGE = model.size_clip

    # we have safety margin along height for coco-search18 dataset
    SIZE_FRAME = (224+params['margin'], 224)

    def displace(img):    
        image = img.resize((224,140))
        image_frame = Image.new("RGB", SIZE_FRAME, (0,0,0))
        image_frame.paste( image, (params['margin'] // 2, (224 - 140) // 2) )

        image = []
        for iw in range(params['margin']):
            for ih in range(params['margin']):
                _le = iw*params['stride']
                _up = ih*params['stride']
                _ri = _le+224
                _lo = _up+224
                image.append( image_frame.crop((_le,_up,_ri,_lo)) )    
                
        return image


# data loader
files = load.load_images(path_data, data_split=data_split, save_images=False)

class ImageDataset(Dataset):
    """Image dataset"""
    def __init__(self, path, high_res):
        self.path = path
        self.high_res = high_res

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        """load images"""
        path = self.path[idx]
        img  = Image.open(path).convert('RGB')
        if self.high_res: 
            img = displace(img)
        else:
            img = [img]
        return img, idx

dataloader = DataLoader(
    ImageDataset(files, high_res), 
    collate_fn=lambda x: x,
    batch_size=batch_size
)

SIZE = (-1, 14*14, 10, 16, 18)
text = list(model.targets.values())


# loop
path_output = Path(path_prcd) / 'maps' / save_name / data_split
path_output.mkdir(parents=True, exist_ok=True)

for i, img in enumerate(dataloader):
    idx = [im[1] for im in img]
    img = sum( [im[0] for im in img], [] )
    
    inputs = processor(
        text=text, images=img, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.get_maps(**inputs)

    # save
    data_files = [Path(f).name for f in files[idx]]
    outputs = [o.detach().cpu().numpy().reshape(SIZE) for o in outputs]

    for ii, fn in enumerate(data_files):        
        with open( path_output / (fn.split('.')[0] + '.pkl'), 'wb' ) as f:
            pickle.dump( [o[ii] for o in outputs], f )
    
    print('complete:', i+1, '/', len(dataloader), 'batches')