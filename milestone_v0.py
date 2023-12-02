import os
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import mmengine
import pandas as pd
from pprint import pprint
import json
from pycocotools.coco import COCO
from tqdm import tqdm
from mmengine import Config
from mmengine.runner import set_random_seed
import random
import torch, torchvision
import mmdet
import mmcv
import mmengine
import subprocess

# Step 0: Get all filenames in the directory
dir_path = "/data/hleeau/imaterialist_checkpoint/mmdetection/imaterialist/train"
filenames = os.listdir(dir_path)

# Step 1: Prepare train_df
# ----------------------------------------------------------------------------------#
# Read the CSV file
file_dir = "mmdetection/imaterialist/train.csv"
train_df = pd.read_csv(file_dir)
# print(filenames)
train_df = train_df[train_df['ImageId'].isin(filenames)]
print(len(train_df))

# Remove all the attributes after "_" in train_df['ClassId']
if "_" in train_df['ClassId'].to_string():
  train_df['ClassId'] = train_df['ClassId'].str.split('_').str.get(0)

# Pre-processing: create new column ImageId
train_df = train_df.rename(columns={'ImageId': 'file_name'})
train_df['ImageId'] = pd.factorize(train_df['file_name'])[0]

def rle_decoding(encoding, height, width):
    """
    Decodes RLE encoding into a binary mask and calculates the masked area and bbox.

    Parameters:
        encoding (str): RLE encoding as a space-separated string.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        mask (1d array): Decoded binary mask with shape (height, width).
        area (float): Total number of pixels in the mask.
        bbox (list): Bounding box coordinates [x, y, width, height].

    """
    # Remove space
    encoding = encoding.split()

    encoding = list(map(int, encoding))

    generated_1d_mask = np.zeros(width*height).astype(int)

    for j in range(len(encoding) // 2):
      generated_1d_mask[encoding[j * 2]:encoding[j * 2]+encoding[j * 2+1]] = 1

    mask = generated_1d_mask.reshape((width, height)).T.astype(int)
    area = float(np.sum(mask))

    # Find the coordinates of the foreground pixels
    coords = np.argwhere(mask == 1)

    # Calculate the bounding box
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    mask = mask.tolist()

    return mask, area, bbox

# Create empty lists for the new columns
segmentation_list = []
area_list = []
bbox_list = []

for i in tqdm(range(len(train_df))):
    encoding = train_df.iloc[i]['EncodedPixels']
    height = train_df.iloc[i]['Height']
    width = train_df.iloc[i]['Width']
    mask, area, bbox = rle_decoding(encoding, height, width)

    # Append the values to the respective lists
    segmentation_list.append(mask)
    area_list.append(area)
    bbox_list.append(bbox)
    # if (i%100 == 0):
    #     print("current:", i)

# Assign the lists to the DataFrame columns
train_df['segmentation'] = segmentation_list
train_df['area'] = area_list
train_df['bbox'] = bbox_list

train_df['iscrowd'] = 0
train_df.drop('EncodedPixels', axis=1, inplace = True)
# ----------------------------------------------------------------------------------#


# Step 2: Prepare coco_base.json
# ----------------------------------------------------------------------------------#
coco_base = { "info": {},
              "images": [],
              "annotations": [],
              "categories": []}

coco_base["info"] = {
    "year": 2019,
    "version": "1.0",
    "description": "The 2019 FGVC^6 iMaterialist Competition - Fashion track dataset.",
    "contributor": "iMaterialist Fashion Competition group",
    "url": "https://github.com/visipedia/imat_comp",
    "date_created": "2019-04-19 12:38:27.493919"
}

category_json = mmengine.load('mmdetection/imaterialist/label_descriptions.json')

coco_base["categories"] = [
    {
        "id": category["id"],
        "name": category["name"],
        "supercategory": category["supercategory"]
    }
    for category in category_json["categories"]
]

# Remove duplicate rows based on the 'id' column
train_df_unique = train_df.drop_duplicates(subset='ImageId')

# Create a dictionary for each unique ID with its corresponding information
unique_ids = train_df_unique.apply(lambda row: {'id': row['ImageId'],
                                                'width': row['Width'],
                                                'height': row['Height'],
                                                'file_name': row['file_name']}, axis=1).tolist()
print("Unique IDs: ", len(unique_ids))
coco_base["images"] = unique_ids

annotation_df = train_df
annotation_df['id'] = range(len(annotation_df))
annotation_df = annotation_df.drop('file_name', axis=1)
annotation_df = annotation_df.drop('Height', axis = 1)
annotation_df = annotation_df.drop('Width', axis = 1)
annotation_df = annotation_df.rename(columns={'ClassId': 'category_id'})
annotation_df = annotation_df.rename(columns={'ImageId': 'image_id'})

annotation_list = annotation_df.to_dict(orient='records')

# Assign parsed JSON object to coco_base["annotations"]
coco_base["annotations"] = annotation_list
# ----------------------------------------------------------------------------------#


# Step 3: Prepare annotation.json & coco_base.json
# ----------------------------------------------------------------------------------#
# Convert the list of dictionaries to a JSON string
# annotation_json = pd.DataFrame(annotation_list).to_json(orient='records')

# # Save the JSON string to a files
# with open('mmdetection/imaterialist/annotation.json', 'w') as f:
#     f.write(annotation_json)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

out_file = open('mmdetection/imaterialist/coco_base100.json','w')
json.dump(coco_base, out_file, cls=NpEncoder)
out_file.close()
# ----------------------------------------------------------------------------------#


# Step 4: Set up configurations
# ----------------------------------------------------------------------------------#
cfg = Config.fromfile('mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py')

category_json = mmengine.load('mmdetection/imaterialist/label_descriptions.json')
class_name = [category["name"] for category in category_json["categories"]]

class_name = tuple(class_name)

# Generate 46 unique RGB color codes
color_codes = set()

while len(color_codes) < 46:
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    color_codes.add((red, green, blue))

color_codes = list(color_codes)

# Modify dataset classes and color
cfg.metainfo = {
    'classes': class_name,
    'palette': color_codes
}

# Modify dataset type and path
cfg.data_root = 'mmdetection/imaterialist'

cfg.train_dataloader.dataset.ann_file = 'coco_base100.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'coco_base100.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'train/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'/'+'coco_base100.json'
# cfg.test_evaluator = cfg.val_evaluators

# Modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 46
cfg.model.roi_head.mask_head.num_classes = 46

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = 'mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './milestone'

# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 3

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optim_wrapper.optimizer.lr = 0.02
cfg.default_hooks.logger.interval = 10


# Set seed thus the results are more reproducible
# cfg.seed = 0
set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

config=f'mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_apparel_milestone.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)
# ----------------------------------------------------------------------------------#


# Step 5: Called tool/train.py file with {config}
# ----------------------------------------------------------------------------------#
subprocess.call(["python", "mmdetection/tools/train.py", config])