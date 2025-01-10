# DTU ML Ops project. Pneumonia detection.
Our project is based on a dataset of chest X-Ray images, targeting the lungs, and aiming to classify them into 2 classes: patients with pneumonia signs, and patients without. Link to the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. The goal is to set up a complete Machine Learning pipeline covering development, deployment, maintenance and operations of a pipeline for detecing pneumonia from x-ray images.

We will be using the PyTorch image models library (timm) for our modelling, and expect to use the PyTorch Lightning framework to remove the need for boilerplate. 

The dataset consists of 5856 .jpeg images, already split into test, train and validation, and normal/pneumonia for each case. 
* Train
Normal → ~1350 images
Pneumonia → ~3890 images
* Test
Normal → ~240 images
Pneumonia → ~400 images
* Validation
Normal → 8 images
Pneumonia → 8 images

Since our dataset of choice contains relatively limited amounts of data, we will try an use some pretrained models, perhaps a resnet variant as a first stab.
