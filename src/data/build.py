"""build data """
import argparse
import load

parser = argparse.ArgumentParser(description='Build data')
parser.add_argument('--data_split', type=str, default='trainval', help='data split')
parser.add_argument('--path_data',  type=str, default='data/external/coco_search18', help='path to the external data')
parser.add_argument('--path_prcd',  type=str, default='data/processed', help='path to save the processed data')

args = parser.parse_args()
data_split = args.data_split
path_data = args.path_data
path_prcd = args.path_prcd

print("Building list of images and indice...")
_, _ = load.load_images(path_data, path_outputs=path_prcd, data_split=data_split)