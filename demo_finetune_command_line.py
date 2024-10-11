import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat,CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)





chat = Chat(model, vis_processor, device=device)

image_path = "../affectnet_test_examples/affectnet_test_examples"
temperature = 1
num_beams = 2
img_list = []
chatbot = None



# ===========================================
#             Modification Pre-prompt
# ===========================================
conv_Example = Conversation(   # Example of a Conversation Object
    system="You are a helpful vision assistant. "
           "You are able to understand the visual content that the user provides.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<s>",
    sep2="</s>",
)
conv_Baseline = CONV_VISION
chat_state = conv_Baseline.copy()




# ===========================================
#             Image Prompts
# ===========================================




# First Template for the Neural Network.
user_message_ListOfFeatures = """"
Eye Color: Describe the color of the character’s eye.
Eye Shape: Talk about the shape of their eyes.
Eyebrows: Mention the thickness, shape, and color of their eyebrows.
Eyelashes: Describe their eyelashes.
Nose Shape: Discuss the shape of their nose.
Mouth/Lips: Describe their lips and the way they move when they speak.
Teeth: If relevant, talk about their teeth.
Skin Tone: Mention their skin color and complexion.
Skin Texture: Discuss any noticeable texture on their skin.
Facial Hair: If applicable, describe their facial hair.
Cheekbones: Talk about the prominence or shape of their cheekbones.
Chin: Describe the shape of their chin.
Forehead: Mention the size and shape of their forehead.
Jawline: Describe their jawline.
Freckles/Moles/Birthmarks: If they have any, talk about the location and appearance of these features.
Expression: Discuss their typical facial expressions or how their face changes with their emotions.
Aging Signs: Mention any signs of aging if relevant.
Symmetry/Asymmetry: Talk about the symmetry or asymmetry of their face.
Ear Shape/Size: Describe the size and shape of their ears.
Facial Proportions: Discuss how the different parts of their face relate to each other in size and placement.
"""

# Secondary prompt
user_message_General_Emotion = """ 
Describe the emotion on the picture.
""" 

# Secondary prompt
user_message_Storytelling = """ 
From the emotion of the character on the picture, imagine a story leading to the present emotion and what would happen after.
""" 

user_message_Captioning = """
    Give 3 or more captions that would fit this picture best.
"""
user_message_Body = """
    From this picture try to guess the pose of the character on the picture. Describe to me the action they may be doing.
"""


####################################################################################################################
#
#  Test Prompt
#
#####################################################################################################################


user_describe_face ="What is on the picture?" # "Could you examine the portrait and describe the shape of the face, the skin tone, and details about the eyes and eyebrows, including their size, shape, and color?\nCan you also provide details on the nose, mouth, ears, and cheekbones, as well as the person's jawline and chin, focusing on their size and shape?\nFinally, could you describe the person's hair, complexion, facial expressions, apparent age, and any makeup, adornments, or facial hair, noting specifics like hair texture, style, and unique facial features?" 
#"Describe this person's facial features and emotions."

###############################################################################

gr_image = image_path


chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."


def upload_image(gr_image=gr_image,chat_state=chat_state,img_list=img_list):
    if gr_image is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_image, chat_state, img_list) # first Chat object function to upload the image with the path
    chat.encode_img(img_list)
    return img_list  # process the image with VIS and puts it on Device 


def prompting(prompt="",chat_state=chat_state,img_list=img_list):   

    
    if prompt:
        user_message = prompt
    chat.ask(user_message, chat_state) # prompting the machine.
    llm_message = chat.answer(conv=chat_state, # Here we get the response from the LLM
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=500, # original has 300 tokens
                                max_length=2000)[0]
    # print("Chat State Prompt:  \n", chat_state.get_prompt())
    # print("Chat State:  \n",chat_state)
    print("LLM Message: \n",llm_message)
    return llm_message


def reset(chat_state,img_list):
    chat_state.messages =[]
    img_list = []
    return chat_state, img_list
    


import os

def list_images(directory, limit=100):
    files = os.listdir(directory)
    #  the files that are images (assuming .jpg, .jpeg, .png extensions for images)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #  the list of image files
    image_files.sort()
    # RE to 'limit' image files
    return image_files[:limit]

#  where the images are stored
directory = '../../Comparison/ffhq'
#  a list of image files
image_list = list_images(directory)


image_path = directory



import json

# Iempty list to hold the objects
data_list = []

if __name__ == "__main__":
    
#  all items in the directory
    all_items = os.listdir(image_path)
    counter = 0
for item in all_items:
    chat_state = conv_Baseline.copy()
    img_list = []
    full_path = os.path.join(image_path, item)
    
    #  if it's a file
    if os.path.isfile(full_path):
        print(full_path,"\n")
        img_list = upload_image(gr_image=full_path,chat_state=chat_state,img_list=img_list)
        # prompting(user_message_ListOfFeatures,chat_state,img_list)
        # prompting(user_message_General_Emotion,chat_state,img_list)
        # prompting(user_message_Storytelling,chat_state,img_list)
        # prompting(user_message_Captioning,chat_state,img_list)
        # prompting(user_message_Body,chat_state,img_list)
        img_list = []
        llm_message = prompting(user_describe_face,chat_state,img_list)

            # C (dictionary in Python) with the path and text
        obj = {
        'path': full_path,  #  file path, replace as needed
        'text': llm_message # Example text, replace as needed
        }
        # pnd the object to the list
        data_list.append(obj)

        counter += 1
        if counter >= 100:
            break

        chat_state, img_list = reset(chat_state, img_list) # resetting before next 


json_filename = 'data_list_original.json'

#  the list of objects to a JSON file
with open(json_filename, 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

print(f'Data saved to {json_filename}')






