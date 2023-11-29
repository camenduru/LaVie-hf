title: LaVie
emoji: 😊
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 4.3.0
app_file: base/app.py
pinned: false

# LaVie: High-Quality Video Generation with Cascaded Latent Diffusion Models

This repository is the official PyTorch implementation of [LaVie](https://arxiv.org/abs/2309.15103).

**LaVie** is a Text-to-Video (T2V) generation framework, and main part of video generation system [Vchitect](http://vchitect.intern-ai.org.cn/).

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2309.15103)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vchitect.github.io/LaVie-project/)
<!--
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)]()
-->

<img src="lavie.gif" width="800">

## Installation
```
conda env create -f environment.yml 
conda activate lavie
```

## Download Pre-Trained models
Download [pre-trained models](https://huggingface.co/YaohuiW/LaVie/tree/main), [stable diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main), [stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/main) to `./pretrained_models`. You should be able to see the following:
```
├── pretrained_models
│   ├── lavie_base.pt
│   ├── lavie_interpolation.pt
│   ├── lavie_vsr.pt
│   ├── stable-diffusion-v1-4
│   │   ├── ...
└── └── stable-diffusion-x4-upscaler
        ├── ...
```

## Inference
The inference contains **Base T2V**, **Video Interpolation** and **Video Super-Resolution** three steps. We provide several options to generate videos:
* **Step1**: 320 x 512 resolution, 16 frames
* **Step1+Step2**: 320 x 512 resolution, 61 frames
* **Step1+Step3**: 1280 x 2048 resolution, 16 frames
* **Step1+Step2+Step3**: 1280 x 2048 resolution, 61 frames

Feel free to try different options:)


### Step1. Base T2V
Run following command to generate videos from base T2V model. 
```
cd base
python pipelines/sample.py --config configs/sample.yaml
```
Edit `text_prompt` in `configs/sample.yaml` to change prompt, results will be saved under `./res/base`. 

### Step2 (optional). Video Interpolation
Run following command to conduct video interpolation.
```
cd interpolation
python sample.py --config configs/sample.yaml
```
The default input video path is `./res/base`, results will be saved under `./res/interpolation`. In `configs/sample.yaml`, you could modify default `input_folder` with `YOUR_INPUT_FOLDER` in `configs/sample.yaml`. Input videos should be named as `prompt1.mp4`, `prompt2.mp4`, ... and put under `YOUR_INPUT_FOLDER`. Launching the code will process all the input videos in `input_folder`.


### Step3 (optional). Video Super-Resolution
Run following command to conduct video super-resolution.
```
cd vsr
python sample.py --config configs/sample.yaml
```
The default input video path is `./res/base` and results will be saved under `./res/vsr`. You could modify default `input_path` with `YOUR_INPUT_FOLDER` in `configs/sample.yaml`. Smiliar to Step2, input videos should be named as `prompt1.mp4`, `prompt2.mp4`, ... and put under `YOUR_INPUT_FOLDER`. Launching the code will process all the input videos in `input_folder`.


## BibTex
```bibtex
@article{wang2023lavie,
  title={LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models},
  author={Wang, Yaohui and Chen, Xinyuan and Ma, Xin and Zhou, Shangchen and Huang, Ziqi and Wang, Yi and Yang, Ceyuan and He, Yinan and Yu, Jiashuo and Yang, Peiqing and others},
  journal={arXiv preprint arXiv:2309.15103},
  year={2023}
}
```

## Acknowledgements
The code is buit upon [diffusers](https://github.com/huggingface/diffusers) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion), we thank all the contributors for open-sourcing. 


## License
The code is licensed under Apache-2.0, model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form]().
