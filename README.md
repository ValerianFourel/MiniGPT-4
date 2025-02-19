# MiniGPTFace

## Training of MiniGPT-4

The training of MiniGPT-4 contains two alignment stages.

**1. First pretraining stage**

In the first pretrained stage, the model is trained using image-text pairs from Laion and CC datasets
to align the vision and language model. To download and prepare the datasets, please check 
our [first stage dataset preparation instruction](dataset/README_1_STAGE.md). 
After the first stage, the visual features are mapped and can be understood by the language
model.
To launch the first stage training, run the following command. In our experiments, we use 4 A100. 
You can change the save path in the config file 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage1_pretrain.yaml)

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

A MiniGPT-4 checkpoint with only stage one training can be downloaded 
[here (13B)](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link) or [here (7B)](https://drive.google.com/file/d/1HihQtCEXUyBM1i9DQbaK934wW3TZi-h5/view?usp=share_link).
Compared to the model after stage two, this checkpoint generate incomplete and repeated sentences frequently.


**2. Second finetuning stage**

In the second stage, we use a small high quality image-text pair dataset created by ourselves
and convert it to a conversation format to further align MiniGPT-4.
To download and prepare our second stage dataset, please check our 
[second stage dataset preparation instruction](dataset/README_2_STAGE.md).
To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

After the second stage alignment, MiniGPT-4 is able to talk about the image coherently and user-friendly. 


## Finetune of MiniGPT-4 Part 2


You firstly need to prepare the dataset. you can follow this step to prepare the dataset.
our [dataset preparation](dataset/README_MINIGPTv2_FINETUNE.md). 

In the train_configs/minigptv2_finetune.yaml, you need to set up the following paths:

llama_model checkpoint path: "/path/to/llama_checkpoint"

ckpt: "/path/to/pretrained_checkpoint"

ckpt save path: "/path/to/save_checkpoint"

For ckpt, you may load from our pretrained model checkpoints:
| MiniGPT-v2 (after stage-2) | MiniGPT-v2 (after stage-3) | MiniGPT-v2 (online developing demo) | 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) |[Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |


```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml
```



<font size='5'>**MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning**</font>


<font size='5'> **MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models**</font>

Deyao Zhu*, Jun Chen*, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny

*equal contribution

*King Abdullah University of Science and Technology*

## Modifications made


We made the following modifications from the original:
Inference files for the command lines:
- demo_v2_command_line.py
- demo_v1_command_line.py

We add dataset files inside of: ~/FaceGPT/Archive/MiniGPT-4/minigpt4/datasets/datasets :
- sharegpt_dataset.py
- chatgpt4vision_datasets.py

We add train configs files inside of: ~/FaceGPT/Archive/MiniGPT-4/train_configs :
- minigpt4_stage2_finetune_gpt4vision.yaml
- minigptv2_finetune_gpt4vision.yaml

We add folders to contain the configurations files for the training files path,
to be found at:  ~/FaceGPT/Archive/MiniGPT-4/minigpt4/configs/datasets/ :
- chatgpt4vision
- sharegpt

Inside of the file MiniGPT-4/minigpt4/datasets/builders/image_text_pair_builder.py :
- we need a wrapper builder class for each of the finetuning dataset (ShareGPT_Face and FaceGPT4Vision) 

We then put of our data, inside of /fast/vfourel/FaceGPT/Data/MiniFaceGPT4Data

For each dataset:
-  we have the ShareGPT Data with images annotated by GPT4Vision and/or the ShareCaptioner by the ShareGPT:
Number of images in each file:
subset_objects_gpt4vision_100k.json: 19514 images
subset_objects_share-captioner_coco_lcs_sam_1246k_1107.json: 17754 images

Number of overlapping image paths: 17759

- We have the GPT4VisionDataset that we have collected by sending images to OpenAI's API for annotations:
We have the subsections:
Object counts in gpt4VisionCalls_Cleaned_BySection.json:
Pre-Prompt: 4 objects
Facial Expression due to contextual cues: 1497 objects
Facial Characteristics Long: 1511 objects
Factual Generalization: 1246 objects
Mouvement Description: 349 objects
Interpretative Generalization: 1346 objects
Intensity of the emotion: 1584 objects
Posture: 739 objects
Race: 1188 objects
Race and Gender: 192 objects
Facial Characteristics Short : 196 objects
Gender: 1205 objects
Human or not: 1182 objects
Feature by Feature Description: 766 objects
Fitzpatrick: 94 objects
Multi-Person Interaction: 0 objects


The train loop is executed inside of MiniGPT-4/minigpt4/task/base_task.py

We modify the files MiniGPT-4/train_configs/minigptv2_finetune_gpt4vision.yaml
to lower the number of workerrs for training to 1, and set the number of
 gradient accumulations from 1 to 8



In the file: MiniGPT-4/minigpt4/models/minigpt_base.py
we modify to         return {"loss": loss,"output s": outputs} # modifications by VF
Such that we can delete outputs to save RAM space
We can then delete it in MiniGPT-4/minigpt4/task/base_task.py
- we modify the function train_step(self, model, samples)
- _train_inner_loop


we try a last thing by adding this as export:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512




## 💡 Get help - [Q&A](https://github.com/Vision-CAIR/MiniGPT-4/discussions/categories/q-a) or [Discord 💬](https://discord.gg/5WdJkjbAeE)

<font size='4'> **Example Community Efforts Built on Top of MiniGPT-4 ** </font> 
  
  <a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2304.10592'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

* <a href='https://github.com/waltonfuture/InstructionGPT-4?tab=readme-ov-file'><img src='https://img.shields.io/badge/Project-Page-Green'></a> **InstructionGPT-4**: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4 Lai Wei, Zihao Jiang, Weiran Huang, Lichao Sun, Arxiv, 2023

* <a href='https://openaccess.thecvf.com/content/ICCV2023W/CLVL/papers/Aubakirova_PatFig_Generating_Short_and_Long_Captions_for_Patent_Figures_ICCVW_2023_paper.pdf'><img src='https://img.shields.io/badge/Project-Page-Green'></a> **PatFig**: Generating Short and Long Captions for Patent Figures.", Aubakirova, Dana, Kim Gerdes, and Lufei Liu, ICCVW, 2023 


* <a href='https://github.com/JoshuaChou2018/SkinGPT-4'><img src='https://img.shields.io/badge/Project-Page-Green'></a> **SkinGPT-4**: An Interactive Dermatology Diagnostic System with Visual Large Language Model, Juexiao Zhou and Xiaonan He and Liyuan Sun and Jiannan Xu and Xiuying Chen and Yuetan Chu and Longxi Zhou and Xingyu Liao and Bin Zhang and Xin Gao,  Arxiv, 2023 


* <a href='https://huggingface.co/Tyrannosaurus/ArtGPT-4'><img src='https://img.shields.io/badge/Project-Page-Green'></a> **ArtGPT-4**: Artistic Vision-Language Understanding with Adapter-enhanced MiniGPT-4.",  Yuan, Zhengqing, Huiwen Xue, Xinyi Wang, Yongming Liu, Zhuanzhe Zhao, and Kun Wang, Arxiv, 2023 


</font>

## News
[Oct.31 2023] We release the evaluation code of our MiniGPT-v2.  

[Oct.24 2023] We release the finetuning code of our MiniGPT-v2.

[Oct.13 2023] Breaking! We release the first major update with our MiniGPT-v2

[Aug.28 2023] We now provide a llama 2 version of MiniGPT-4

## Online Demo

Click the image to chat with MiniGPT-v2 around your images
[![demo](figs/minigpt2_demo.png)](https://minigpt-v2.github.io/)

Click the image to chat with MiniGPT-4 around your images
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)


## MiniGPT-v2 Examples

![MiniGPT-v2 demos](figs/demo.png)



## MiniGPT-4 Examples
  |   |   |
:-------------------------:|:-------------------------:
![find wild](figs/examples/wop_2.png) |  ![write story](figs/examples/ad_2.png)
![solve problem](figs/examples/fix_1.png)  |  ![write Poem](figs/examples/rhyme_1.png)

More examples can be found in the [project page](https://minigpt-4.github.io).



## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
```


**2. Prepare the pretrained LLM weights**

**MiniGPT-v2** is based on Llama2 Chat 7B. For **MiniGPT-4**, we have both Vicuna V0 and Llama 2 version.
Download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

|                            Llama 2 Chat 7B                             |                                           Vicuna V0 13B                                           |                                          Vicuna V0 7B                                          |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
[Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) | [Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) 


Then, set the variable *llama_model* in the model config file to the LLM weight path.

* For MiniGPT-v2, set the LLM path 
[here](minigpt4/configs/models/minigpt_v2.yaml#L15) at Line 14.

* For MiniGPT-4 (Llama2), set the LLM path 
[here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.

* For MiniGPT-4 (Vicuna), set the LLM path 
[here](minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) at Line 18

**3. Prepare the pretrained model checkpoints**

Download the pretrained model checkpoints


| MiniGPT-v2 (after stage-2) | MiniGPT-v2 (after stage-3) | MiniGPT-v2 (online developing demo)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) |[Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |


For **MiniGPT-v2**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigptv2_eval.yaml](eval_configs/minigptv2_eval.yaml#L10) at Line 8.



| MiniGPT-4 (Vicuna 13B) | MiniGPT-4 (Vicuna 7B) | MiniGPT-4 (LLaMA-2 Chat 7B) |
|----------------------------|---------------------------|---------------------------------|
| [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) | [Download](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing) |

For **MiniGPT-4**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 8 for Vicuna version or [eval_configs/minigpt4_llama2_eval.yaml](eval_configs/minigpt4_llama2_eval.yaml#L10) for LLama2 version.   



### Launching Demo Locally

For MiniGPT-v2, run
```
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0
```

For MiniGPT-4 (Vicuna version), run

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

For MiniGPT-4 (Llama2 version), run

```
python demo.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```


To save GPU memory, LLMs loads as 8 bit by default, with a beam search width of 1. 
This configuration requires about 23G GPU memory for 13B LLM and 11.5G GPU memory for 7B LLM. 
For more powerful GPUs, you can run the model
in 16 bit by setting `low_resource` to `False` in the relevant config file:

* MiniGPT-v2: [minigptv2_eval.yaml](eval_configs/minigptv2_eval.yaml#6) 
* MiniGPT-4 (Llama2): [minigpt4_llama2_eval.yaml](eval_configs/minigpt4_llama2_eval.yaml#6)
* MiniGPT-4 (Vicuna): [minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#6)

Thanks [@WangRongsheng](https://github.com/WangRongsheng), you can also run MiniGPT-4 on [Colab](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)


### Training
For training details of MiniGPT-4, check [here](MiniGPT4_Train.md).

For finetuning details of MiniGPT-v2, check [here](MiniGPTv2_Train.md)


### Evaluation
For finetuning details of MiniGPT-v2, check [here](eval_scripts/EVAL_README.md)  


## Acknowledgement

+ [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) The model architecture of MiniGPT-4 follows BLIP-2. Don't forget to check this great open-source work if you don't know it before!
+ [Lavis](https://github.com/salesforce/LAVIS) This repository is built upon Lavis!
+ [Vicuna](https://github.com/lm-sys/FastChat) The fantastic language ability of Vicuna with only 13B parameters is just amazing. And it is open-source!
+ [LLaMA](https://github.com/facebookresearch/llama) The strong open-sourced LLaMA 2 language model.


If you're using MiniGPT-4/MiniGPT-v2 in your research or applications, please cite using this BibTeX:
```bibtex


@article{chen2023minigptv2,
      title={MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning}, 
      author={Chen, Jun and Zhu, Deyao and Shen, Xiaoqian and Li, Xiang and Liu, Zechu and Zhang, Pengchuan and Krishnamoorthi, Raghuraman and Chandra, Vikas and Xiong, Yunyang and Elhoseiny, Mohamed},
      year={2023},
      journal={arXiv preprint arXiv:2310.09478},
}

@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
```


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).
