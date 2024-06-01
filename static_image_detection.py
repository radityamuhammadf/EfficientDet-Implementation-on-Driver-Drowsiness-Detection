import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

compound_coef = 0
force_input_size = None  # set None to use default size

img_path = r'D:\01-KULIAH\0-SEMESTER 8\01-Undergraduate Thesis\0-Laboratory\EfficientDet Inference\Yet-Another-EfficientDet-Pytorch\test\test\data_0346_png.rf.81e77364efb0f31a6d3ea2258846e84e.jpg'

threshold = 0.2
iou_threshold = 0.2

use_cuda = torch.cuda.is_available()  # Modified to check if CUDA is available
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['closed-eyes', 'yawn']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

# Loading the model on CPU if CUDA is not available
if use_cuda:
    model_path = r"D:\01-KULIAH\0-SEMESTER 8\01-Undergraduate Thesis\0-Laboratory\EfficientDet Inference\Yet-Another-EfficientDet-Pytorch\logs\blink_yawn\efficientdet-d0_14_9570.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
else:
    model_path = r"D:\01-KULIAH\0-SEMESTER 8\01-Undergraduate Thesis\0-Laboratory\EfficientDet Inference\Yet-Another-EfficientDet-Pytorch\logs\blink_yawn\efficientdet-d0_14_9570.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.requires_grad_(False)
model.eval()

if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

out = invert_affine(framed_metas, out)

for i in range(len(ori_imgs)):
    if len(out[i]['rois']) == 0:
        continue
    ori_imgs[i] = ori_imgs[i].copy()
    for j in range(len(out[i]['rois'])):
        (x1, y1, x2, y2) = out[i]['rois'][j].astype(int)  # Changed np.int to int
        cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
        obj = obj_list[out[i]['class_ids'][j]]
        score = float(out[i]['scores'][j])

        cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
    
    # Display the image with detections
    cv2.imshow('Static Image Detection Test', ori_imgs[i])

# Wait for a key press indefinitely or for a specific amount of time
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
