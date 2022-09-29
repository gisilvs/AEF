import argparse
import os

import numpy as np
from PIL import Image
import pandas as pd


def resize_celebahq(data_folder, output_folder, output_size, dir_image_list='image_list.txt', dir_eval_partition='list_eval_partition.txt'):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    if not os.path.isdir(os.path.join(output_folder, 'train')):
        os.mkdir(os.path.join(output_folder, 'train'))
    if not os.path.isdir(os.path.join(output_folder, 'valid')):
        os.mkdir(os.path.join(output_folder, 'valid'))
    if not os.path.isdir(os.path.join(output_folder, 'test')):
        os.mkdir(os.path.join(output_folder, 'test'))


    df_img_list = pd.read_csv(dir_image_list, sep="\s+")
    df_eval_partition = pd.read_csv(dir_eval_partition, sep=" ", names=['image', 'split'])
    image_folder = data_folder
    for i, row in df_eval_partition.iterrows():
        if not df_img_list[df_img_list['orig_file'] == row['image']].empty:
            img_name = str(df_img_list[df_img_list['orig_file'] == row['image']]['idx'].iloc[0] + 1).zfill(5)
            image = np.asarray(Image.open(os.path.join(image_folder, f'{img_name}.jpg')).resize((output_size, output_size)))
            if row['split'] == 0:
                dest = 'train'
            elif row['split'] == 1:
                dest = 'valid'
            elif row['split'] == 2:
                dest = 'test'
            np.savez(os.path.join(output_folder, dest, f'{img_name}.npz'), image=image)

parser = argparse.ArgumentParser(description='CelebA-HQ resizing tool')
parser.add_argument('--data-dir', type=str, help='location of the original CelebA-HQ folder')
parser.add_argument('--output-folder', type=str, default='celebahq', help='location of the output folder')
parser.add_argument('--dimension', type=int, default=32, help='new dimensionality of the image, i.e. dim x dim')

args = parser.parse_args()
resize_celebahq(args.data_dir, args.output_folder, args.dimension)