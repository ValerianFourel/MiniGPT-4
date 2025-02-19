# /fast/vfourel/FaceGPT/Data/StableFaceData/AffectNet41k_FlameRender_Descriptions_Images/affectnet_41k_AffectOnly/Modified_PromptsSmall_AffectOnly_41ksamples.json


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


###############################################################################################################
#
# By Valerian FOUREL. This is a simple refitting of the LLaVADetail dataset for the 
# Data Generated by the ShareGPT Team with GPT4 vision. We used the same template of 
# dataset as LLaVA as the dataset have the same structure.
#
################################################################################################################
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class RealisticEmotionsDetailDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

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
