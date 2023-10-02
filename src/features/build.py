"""building predictive maps"""
# TODO: z-scoring of the human consistency
import argparse


import time
import numpy as np
from scipy.stats import zscore

# Given name of the model, build predictive maps
parser = argparse.ArgumentParser(description='Build predictive maps')
parser.add_argument('--model', type=str, default='smm', help='model name')

# TODO: change path_images to path_processed
parser.add_argument('--path_images', type=str, default='data/processed/images', help='path to the images')

args = parser.parse_args()
model_name = args.model
path_images = args.path_images


# build smm
if model_name == 'smm':
    pass

# build itti-koch
if model_name == 'ittikoch':
    pass

# build deepgaze-IIE
if model_name == 'deepgaze':
    pass

# build human consistency
if model_name == 'human':
    pass




 
cache_dir  = "/home/groups/jlg/hwgu/checkpoints/huggingface/hub/"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
model.to(DEVICE) # send the model to the chosen device ('cpu' or 'cuda')
model.eval()     # set the model to evaluation mode, since you are not training it

# 
with torch.no_grad():
    W = model.visual_projection.weight
    W = W.detach().cpu().numpy().T

# extract image features
batch_size = 20
output_base = '/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/final'

with open('/scratch/users/hwgu/data/coco/coco.pkl', 'rb') as f:
    images = pickle.load(f)

data_loader = DataLoader(
    ImageDataset(transform),
    collate_fn=collate_fn,
    batch_size=batch_size
)

for i, d in tqdm(enumerate(data_loader), total=len(data_loader)):
    filename = os.path.join(output_base, f'act_{i:04d}.pkl')

    with torch.no_grad():
        image_tokens = processor.feature_extractor(images=d, return_tensors="pt")
        image_tokens['pixel_values'] = image_tokens['pixel_values'].to(device)

        out = model.vision_model(**image_tokens, output_hidden_states=True)

        with open(filename, 'wb') as f:
            pickle.dump({
                'pooler_output': out['pooler_output'].detach().cpu().numpy(),
                'hidden_states': [o.detach().cpu().numpy() for o in out['hidden_states']]
            }, f)


# build text features
tasks = np.unique([sc['task'] for sc in scanpaths])
tasks = [f'a photo of a {task}' for task in tasks]
tasks[12] = 'a photo of an oven'
list_tasks = np.unique([sc['task'] for sc in scanpaths])

task_tokens = []
for task in tasks:
    task_token = processor.tokenizer(text=[task], return_tensors="pt")
    task_token['input_ids'] = task_token['input_ids'].to(device)
    task_token['attention_mask'] = task_token['attention_mask'].to(device)

    with torch.no_grad():
        task_outputs = model.text_model(**task_token)
        task_embeds  = model.text_projection(task_outputs[1])
    task_tokens.append(task_embeds)
task_tokens  = torch.concat(task_tokens, dim=0).detach().cpu().numpy()
task_tokens /= np.sqrt(np.sum(task_tokens**2,axis=-1, keepdims=-1))
with torch.no_grad():
    logit_scale = model.logit_scale.exp().detach().cpu().numpy()



# 
act_list = np.sort(os.listdir('/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/final_trainval/'))

outs = []
for i_act, v_act in enumerate(act_list):
    with open(f'/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/final_trainval/{v_act}', 'rb') as f:
        out_pickle = pickle.load(f)
    
    out_layer = []
    for i_l, v_l in enumerate(np.arange(0,25,1)):
        
        # [batch_size, seq_len, hidden_d]
        hidden_states  = out_pickle['hidden_states'][v_l]

        # project to the comparable state
        hidden_states  = hidden_states @ W

        # spatial information
        hidden_states  = hidden_states[:, 1:, :] 

        # cross-attention with task vector
        hidden_states  = hidden_states @ task_tokens.T
        out_layer.append(hidden_states)
        
    outs.append(out_layer)
    print(v_act)
    
outs = np.concatenate([np.stack(o) for o in outs],axis=1)
with open(f'/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/tmpdir/linear/final_trainval.pkl', 'wb') as f:
    pickle.dump(outs, f)

#
# TODO: THIS COULD BE MUCH SIMPLER

#
def semantic_matching(outs, img, tgt):
    
    # taking softmax
    tr = F.softmax(torch.tensor(
        np.stack([
            np.mean(-outs[:,img,:,:],axis=-1),
            -outs[:,img,:,tgt],
        ],axis=-1)), dim=-1).numpy()[:,:,-1]
    
    # truncate the other parts [THIS IS IMPORTANT]
    tr = tr.reshape((-1,16,16))
    tr = tr[:,3:13,:] 
    
    return tr


# 
def save_maps():
    pass
# concatenate 

act_list = np.sort(os.listdir('/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/final_trainval/'))

# Saving for scoring based on standard measures
# {'name_img': template_response [n_layer x 10 x 16 x n_task]}

res = {}
for i_coco, v_coco in enumerate(coco_index):
    _tr = np.nan*np.zeros([25,10,16,18])
    for i_task in range(18):
        _tr[:,:,:,i_task] = template_response(outs,i_coco,i_task,z_score=False)
        res[v_coco] = _tr
        
# base_dir = '/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/tmpdir'
base_dir = '/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/tmpdir/linear'
base_dir = '/scratch/users/hwgu/data/coco/clip_vit_l_14/whole/tmpdir/random'
with open(f'{base_dir}/search_trainval.pkl', 'wb') as f:
    pickle.dump(res, f)