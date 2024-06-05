import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from torch.profiler import profile, record_function, ProfilerActivity
import os

current_directory = os.getcwd()
# Video's path
video_src=os.path.join(current_directory, r"test_video\10-MaleGlasses-Trim.avi")
# video_src = 0  # set int to use webcam, set str to read from a video file

compound_coef = 0
force_input_size = None  # set None to use default size

threshold = 0.6 #matched with YOLO's confidence score threshold
iou_threshold = 0.5 #matched with YOLO's IoU Threshold

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

def main():
    obj_list = ['closed-eyes', 'yawn']

    # Initialize the dictionary to keep track of detection times and last durations
    detections = {
        'closed-eyes': {'duration': 0, 'last_duration': 0, 'last_seen': None},
        'yawn': {'duration': 0, 'last_duration': 0, 'last_seen': None}
    }

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model_path = os.path.join(current_directory,r"saved_weights\efficientdet-d0_14_9500.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # function for display
    def display(preds, imgs):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                return imgs[i]

            for j in range(len(preds[i]['rois'])):
                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
            
            return imgs[i]

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            with torch.no_grad():
                with record_function("model_inference"):
                    features, regression, classification, anchors = model(x)

                with record_function("postprocess"):
                    out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

                out = invert_affine(framed_metas, out)
                #olah data out (pahami dia tensor seperti apa dan apakah bisa detect class nya, kemudian implementasikan logic yolo ke sini)
                for detection_result in out:
                    detected_class=detection_result['class_ids']
                    

                img_show = display(out, ori_imgs)
            

            cv2.imshow('frame', img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__='__main__':
    main()
# Print profiling results
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
