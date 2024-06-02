import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_video, invert_affine, postprocess
import openvino as ov
from torch.profiler import profile, record_function, ProfilerActivity
import os

current_directory=os.getcwd()

compound_coef = 0
force_input_size = None  # set None to use default size

threshold = 0.6
iou_threshold = 0.6

#check CUDA availability
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Current CUDA device:", torch.cuda.current_device())


use_cuda = torch.cuda.is_available()
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['closed-eyes', 'yawn']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
# input_size = 640

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

model_path=os.path.join(current_directory,r"saved_weights\efficientdet-d0_14_9500.pth")
if use_cuda:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.requires_grad_(False)
model.eval()
# ov_model = ov.convert_model(model)
# core=ov.Core()
# compiled_model = core.compile_model(ov_model) # Compile model from memory

# if use_float16:
#     model = model.half()

# Initialize webcam
# cap = cv2.VideoCapture(0)  # 0 is the default webcam
video_path=os.path.join(current_directory,r"test_video\10-MaleGlasses.avi")
cap = cv2.VideoCapture(video_path)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with record_function("preprocess_video"):
            # Preprocess frame
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

        with record_function("convert_to_tensors"):
            # Converting frames to tensors
            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            with record_function("model_inference"):
                features, regression, classification, anchors = model(x)

            with record_function("postprocess"):
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

        with record_function("invert_affine"):
            out = invert_affine(framed_metas, out)

        with record_function("visualization"):
            for i in range(len(ori_imgs)):
                if len(out[i]['rois']) == 0:
                    continue
                ori_imgs[i] = ori_imgs[i].copy()
                for j in range(len(out[i]['rois'])):
                    (x1, y1, x2, y2) = out[i]['rois'][j].astype(int)
                    cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                    obj = obj_list[out[i]['class_ids'][j]]
                    score = float(out[i]['scores'][j])

                    cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                                (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)

                # Display the result
                cv2.imshow('Live Detection', ori_imgs[i])

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

cap.release()
cv2.destroyAllWindows()

# Print profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
