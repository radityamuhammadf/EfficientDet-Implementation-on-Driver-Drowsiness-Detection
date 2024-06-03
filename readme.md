# EfficientDet Implementation on Driver Drowsiness Detection

An implementation of [Yet Another EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/tree/master) repository for training on blink and yawn datasets to detect driver drowsiness.

## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)

## About
This inference repository uses the transfer learning method from EfficientDet-D0 with the following training configuration:
- Efficientnet-b0 backbone
- Image input size: 512
- Learning rate: 0,001 (1e-3)
- Batch size: 16
- Epochs: 15
- Dataloader workers: 2 

The training was stopped at 15 epochs due to an increasing total loss (it's unclear if this is relevant). The model was tested with 1,447 images (10% of the dataset) and resulted in the following metrics: 
```py
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.675
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.945
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.832
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.675
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.756
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.756
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.756
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
```

## Installation
1. Clone this GitHub repository:
```bash
git clone https://github.com/radityamuhammadf/EfficientDet-Implementation-on-Driver-Drowsiness-Detection.git
```
2. Change the current working directory (cwd):
```bash
cd EfficientDet-Implementation-on-Driver-Drowsiness-Detection
```
3. Create a virtual environment:
```sh
python -m venv venv
```
4. Activate virtual environment:
```sh
python -m venv venv
```
5. Install all the dependencies
Note: Ensure your PyTorch configuration matches your CUDA version (e.g., if you're using CUDA 12.5, consider using [PyTorch for CUDA 12.4](https://pytorch.org/) ) 
```sh
pip install -r requriements.txt
```
6. Manually download the test videos and manually make some folder named `test_videos`, paste the downloaded video in there.
[Video Download Link (GDrive)](https://drive.google.com/drive/folders/1LRUUzz8F_V_rCJAqRk9b2W4nyrwJ_udg?usp=sharing)

## Usage
Run the inference code
```sh
py video_input.py
```
**Additional Information**
Some modules may not be listed in the requirements.txt file (it is unclear why they are missing even after using the pip freeze command). If you encounter a `ModuleNotFoundError`, you can install the missing module using the pip command.


