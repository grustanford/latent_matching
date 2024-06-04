"""make semantic matching maps"""
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
parser.add_argument('--batch_size',  type=int, default=4, help='batch size')
parser.add_argument('--super-res',   type=float, default=1, help='resolution compared to the patch size')

args = parser.parse_args()
data_split  = args.data_split
path_prcd   = args.path
path_data   = args.path_data
model_name  = args.model_name
custom_name = args.custom_name
save_name   = args.save_name
cache_dir   = args.cache_dir
super_res   = args.super_res
batch_size  = args.batch_size

if custom_name is None:
    custom_name = model_name

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
from smm import SemanticMatchingModel, SemanticMatchingProcessor
model = SemanticMatchingModel.from_pretrained(custom_name).to(device)
processor = SemanticMatchingProcessor.from_pretrained(custom_name)
model.eval()
with torch.no_grad(): 
    model.set_origin(processor)  # set the origin of target vectors
print('Semantic matching model loaded, using device:', device)


# data loader
files = load.load_images(path_data, data_split=data_split, save_images=False)

data_dict, sbj_dict = load.load_indices(
    path_prcd     = path_prcd, 
    path_scanpath = path_data, 
    data_split    = data_split,
    merge         = True,
)

class ImageDataset(Dataset):
    def __init__(self, files, data_dict, list_tasks):
        self.files = files
        self.idx_eval = data_dict['idx_eval']
        self.list_tasks = list_tasks

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        fname = Path(fpath).name
        img = Image.open(fpath)
        
        # todo : allow other image size
        padded_img = Image.new("RGB", (1680, 1680), (0, 0, 0))
        padded_img.paste(img, (0, 315))

        # list of tasks corresponding to the image 
        tasks = [self.list_tasks[t] for t in self.idx_eval[:,1][self.idx_eval[:,0]==idx]]

        return padded_img, tasks, fname

list_tasks = list(model.targets.keys())
dataloader = DataLoader(
    ImageDataset(files, data_dict, list_tasks), 
    batch_size=batch_size, 
    collate_fn=lambda x: x
)


# loop
path_output = Path(path_prcd) / 'maps' / save_name / data_split
path_output.mkdir(parents=True, exist_ok=True)

for i, batch in enumerate(dataloader):
    images, tasks, filenames = zip(*batch)
    inputs = processor(
        text=tasks, images=images, return_tensors="pt", padding=True, super_resolution=super_res,
    ).to(device)

    with torch.no_grad():
        outputs = model.get_maps(**inputs)
        outputs = processor.post_process(outputs, super_resolution=super_res, slices=model.slices)

    # Save results
    for fn, tas, out in zip(filenames, tasks, outputs):
        res = {t: o for t, o in zip(tas, out)}
        with open(path_output / (fn.split('.')[0] + '.pkl'), 'wb') as f:
            pickle.dump(res, f)
    
    print('complete:', i + 1, '/', len(dataloader), 'batches')