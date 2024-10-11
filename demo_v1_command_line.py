import argparse
import os
import random
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ========================================
#             Launch Arguments
# ========================================

# python demo_v1_command_line.py --cfg-path eval_configs/minigpt4_eval.yaml --image /home/vfourel/FaceGPT/Comparison/ffhq2/01217.png

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--image", required=True, help="path to input image")
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================



conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

class MiniGPT4CLI:
    def __init__(self, cfg_path, gpu_id, options):
        self.cfg_path = cfg_path
        self.gpu_id = gpu_id
        self.options = options
        self.chat_state = None
        self.img_list = None
        
        # Here you would initialize your model using the provided configuration
        # For example:
        # self.model = initialize_model(cfg_path, gpu_id, options)
        
    def reset(self):
        self.chat_state = None
        self.img_list = None

    def upload_img(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return False

        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return False

        self.chat_state = CONV_VISION.copy()
        self.img_list = []
        llm_message = chat.upload_img(img, self.chat_state, self.img_list)
        chat.encode_img(self.img_list)
        print("Image uploaded successfully.")
        return True

    def ask(self, user_message):
        if not self.chat_state:
            print("Please upload an image first.")
            return

        chat.ask(user_message, self.chat_state)
        print(f"User: {user_message}")

    def answer(self, num_beams=1, temperature=1.0):
        if not self.chat_state:
            print("Please upload an image and ask a question first.")
            return

        llm_message = chat.answer(
            conv=self.chat_state,
            img_list=self.img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=300,
            max_length=2000
        )[0]
        print(f"MiniGPT-4: {llm_message}")

def main():
    args = parse_args()
    
    minigpt4 = MiniGPT4CLI(args.cfg_path, args.gpu_id, args.options)

    if not minigpt4.upload_img(args.image):
        return

    print("Image uploaded. You can start chatting now.")
    print("Type 'quit' to exit the program.")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == 'quit':
            break

        minigpt4.ask(user_input)
        minigpt4.answer()

    print("Thank you for using MiniGPT-4!")

if __name__ == "__main__":
    main()