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


def main():
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
    obj_list = ['closed-eyes', 'yawn']


    # Initialize the dictionary to keep track of detection times and last durations
    detections = {
        'closed-eyes': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None},
        'yawn': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None}
    }

    drowsy_state = False
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

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)
    # Frame per Second variable initiation
    fps=cap.get(cv2.CAP_PROP_FPS)
    frame_duration=1/fps
    frame_number=0

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

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
                    out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                                      iou_threshold)

                out = invert_affine(framed_metas, out)

                current_detections = set()
                for i in range(len(ori_imgs)):
                    for j in range(len(out[i]['rois'])):
                        (x1, y1, x2, y2) = out[i]['rois'][j].astype(int)
                        cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                        obj = obj_list[out[i]['class_ids'][j]]
                        score = float(out[i]['scores'][j])

                        cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 1)

                        # recording the frame count
                        class_name = obj  # reusing the obj variable to match other repo's paradigm
                        if detections[class_name]['last_seen_frame'] is None:
                            detections[class_name]['last_seen_frame'] = frame_number
                        else:
                            detections[class_name]['frame_count'] += frame_number - detections[class_name][
                                'last_seen_frame']
                            detections[class_name]['last_seen_frame'] = frame_number
                        detections[class_name]['was_detected'] = True  # params to state the detected state

                        current_detections.add(class_name)

                # Reset the durations when the class are not detected anymore
                for class_name in detections:
                    if class_name not in current_detections:  # check whether this current frame still detecting the same class
                        detections[class_name]['was_detected'] = False  # reset the previously detected class state
                        detections[class_name]['frame_count'] = 0  # reset counted frame value
                        detections[class_name]['last_seen_frame'] = None  # reset previously seen frame

                # Convert frame counts to time using FPS
                for class_name in detections:
                    detections[class_name]['duration'] = detections[class_name]['frame_count'] * frame_duration

                # Detect drowsiness based on the duration of closed-eyes and yawn
                closed_eyes_duration = detections['closed-eyes']['duration']
                yawn_duration = detections['yawn']['duration']

                # Logic for detecting drowsiness
                if closed_eyes_duration > 2.0 or yawn_duration > 1.5:  # thresholds in seconds
                    drowsy_state = True
                else:
                    drowsy_state = False

            # debugging print state
            print(f"Closed-eyes duration: {closed_eyes_duration:.2f} seconds")
            print(f"Yawn duration: {yawn_duration:.2f} seconds")
            print(f"Drowsy state: {drowsy_state}")

            # Drowsy State Branch Logic
            if drowsy_state is True:
                pass



            """
            Static Information Display Function
            """
            #drawing and writing the annotation
            cv2.rectangle(frame, (0, 10), (200, 60), (255, 255, 255), -1)
            y_offset = 30
            cv2.putText(frame, f'closed-eyes: {closed_eyes_duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
            y_offset += 20
            cv2.putText(frame, f'yawn: {yawn_duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
            y_offset += 20

            cv2.imshow('Inference-EfficientDetD0', ori_imgs[i])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
# Print profiling results
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
