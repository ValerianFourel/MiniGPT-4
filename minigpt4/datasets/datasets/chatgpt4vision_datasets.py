import os
import json
import pickle
import random
import time
# import iterto
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset



###############################################################################################################
#
# By Valerian FOUREL. We are now developing the dataset object for the data collected using GPT4 Vision's API
# We must readapt it in to get what we need 
# 
# 
#
################################################################################################################



class Gpt4VisionFaceDetailDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path,subsets_prompts):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.subsets_prompts = subsets_prompts

        with open(ann_path, 'r') as f:
            data = json.load(f)
            concatenated_objects = []

            #  through the keys in subsets_prompts and concatenate their lists of objects
            for key in self.subsets_prompts :
                if key in data and isinstance(data[key], list):
                    concatenated_objects.extend(data[key])
            self.ann = concatenated_objects

            

    def __len__(self):
        

        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['image']
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['conversations'][1]['value']
        instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        
        instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(instruction))

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['id'],
        }
