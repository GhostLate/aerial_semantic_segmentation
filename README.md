# Aerial semantic segmentation

This repository presents approach for semantic segmentation using [Semantic Drone Dataset](https://www.tugraz.at/index.php?id=22387) dataset. 
You can also find this dataset on [Kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)

# What is purpose of this repo?

This repo aims to do experiments and verify the idea of robust semantic segmentation and specific datasets.

# Semantic Droone Dataset

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
