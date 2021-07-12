# Aerial semantic segmentation

This repository presents approach for semantic segmentation using [Semantic Drone Dataset](https://www.tugraz.at/index.php?id=22387) dataset. 
You can also find this dataset on [Kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)

### What is purpose of this repo?

This repo aims to do experiments and verify the idea of robust semantic segmentation and specific datasets.

### Semantic Droone Dataset

The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. 
The imagery depicts  more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. 

Original images and labels had resolution 6000x4000px (24Mpx). They were reduced to 768x512px size.
The training set contains 300 and the test set is made up of 100 images. 

The original Semantic Drone Dataset contents 24 semanic classes of:
- `background` - `paved-area` - `dirt` - `grass`
- `gravel` - `water` - `rocks` - `pool` - `tree`  
- `vegetation` - `roof` - `wall` - `window`
- `door` - `fence` - `fence-pole` - `person`
- `dog` - `car` - `bicycle` - `conflicting`
- `bald-tree` - `ar-marker` - `obstacle`

After uniting some classes to more general view, 24 classes was reduced to 12:
- `road` - `ground` - `water` - `person` - `car`
- `vegetation` - `construction` - `bicycle` 
- `dog` - `obstacle` - `conflicting` - `background`

### Checkpoints

| Method | Backbone | pretrain img size | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU(ms+flip) | Num Clasess | config | download |
| ------ | -------- | ----------------- | --------- | ---------- | ------- | -------- | ------------- | ----------- | ------ | -------: | 
| UperNet | Swin-T  | 768x512           | 384x384   | 4          | 160000  | -        | -             | 12          | [config](https://github.com/GhostLate/aerial_semantic_segmentation/blob/main/configs/config_SWIN_12.py) | [model](https://drive.google.com/file/d/14Qb9MrC-MJJKaDHnaa0IiKjRoFTqxwGW/view?usp=sharing) &#124; [log]() |
| OCRNet | HRNetV2p-W48 | 768x512       | 384x384   | 2          | 160000  | -        | -             | 12          | [config](https://github.com/GhostLate/aerial_semantic_segmentation/blob/main/configs/config_HRNetV2_W48.py) | [model](https://drive.google.com/file/d/165AHie9s60gFLi-aRHADIXAaJBhKzIAk/view?usp=sharing) &#124; [log]() |
| DNL    | ResNet-101-D8 | 768x512      | 512x512   | 2          | 80000   | -        | -             | 12          | [config](https://github.com/GhostLate/aerial_semantic_segmentation/blob/main/configs/config_ResNet.py) | [model](https://drive.google.com/file/d/10JERyy_UF80bo9E2-48IsN3qi2-vKb4r/view?usp=sharing) &#124; [log]() |
| UperNet | Swin-T  | 768x512           | 384x384   | 4          | 160000  | -        | -             | 24          | [config](https://github.com/GhostLate/aerial_semantic_segmentation/blob/main/configs/config_SWIN_24.py) | [model](https://drive.google.com/file/d/10nqYeUiWfZ6zvGz6QGDWSgnzzP144SEr/view?usp=sharing) &#124; [log]() |

### Usage

`inference.py` allow you to test models on images/videos in dir you choose. Control keys are: 
- `a` - left image/video file in dir 
- `d` - right image/video file in dir
- `w` - turn on/off image/video segmentation
- `s` - save frame in dir
