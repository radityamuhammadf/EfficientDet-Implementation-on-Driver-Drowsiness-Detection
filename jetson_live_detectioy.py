import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
import os
import simpleaudio as sa
from pydub import AudioSegment
import threading

def play_audio(audio):
    audio_obj = audio.play()
    audio_obj.wait_done()


def main():
    current_directory = os.getcwd()
    # Video's path
    # video_src=os.path.join(current_directory, r"test_video\10-MaleGlasses-Trim.avi")
    video_src = 0  # set int to use webcam, set str to read from a video file

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
    # Initialize so-called tracking logic dictionary -> for recording latest detection if there's no detection
    prev_annotation={
        'inference_results':[],
        'rep_count':12 # -> delaying maximum 12 frames if the current frame don't have any detection result (12 frames equal to 0.36 [by 12*0.03])
    }

    drowsy_state = False
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model_path = os.path.join(current_directory,r"saved_weights/efficientdet-d0_14_9500.pth")
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
    print("FPS: ",fps)

    # Warning Sound
    # Initialization - Load the audio file and convert to wav
    audio = AudioSegment.from_mp3(os.path.join(current_directory,"audio/warning.mp3"))
    audio.export("audio/warning.wav", format="wav")

    # Load the WAV file
    warning_sound = sa.WaveObject.from_wave_file("audio/warning.wav")

    drowsy_state = False

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
            features, regression, classification, anchors = model(x)
            out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                              iou_threshold)

        out = invert_affine(framed_metas, out)
        # Check if any detections were made
        if not any(len(detection['rois']) > 0 for detection in out):
            if prev_annotation['rep_count'] > 0 and prev_annotation['inference_results']:
                features, regression, classification, anchors = prev_annotation['inference_results']
                out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                                iou_threshold)
                out = invert_affine(framed_metas, out)
                print("DEBUG- using previously detected annotation")
                prev_annotation['rep_count'] -= 1
        else: #check if there's detection been made
            prev_annotation['inference_results'] = [features, regression, classification, anchors]
            prev_annotation['rep_count'] = 12

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
            # this default frames per second are 30FPS -> and the duration for each frame are 1/30 ~ 0.33
            # for instance, if the closed-eyes class are detected for 65 frames consecutively
            # that means the durations of detected closed-eyes are 65*0.03 which are 1.95s
            detections[class_name]['duration'] = detections[class_name]['frame_count'] * frame_duration

        # Detect drowsiness based on the duration of closed-eyes and yawn
        closed_eyes_duration = detections['closed-eyes']['duration']
        yawn_duration = detections['yawn']['duration']

        # Logic for detecting drowsiness
        if closed_eyes_duration > 0.4 or yawn_duration > 5.0:  # thresholds in seconds
            drowsy_state = True
        else:
            drowsy_state = False

        # Drowsy State Branch Logic
        if drowsy_state is True:
            cv2.rectangle(frame, (500, 20), (640, 60), (255, 255, 255), -1)
            cv2.putText(frame, 'Drowsy', (500, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        # debugging print state
        print(f"Closed-eyes duration: {closed_eyes_duration:.2f} seconds")
        print(f"Yawn duration: {yawn_duration:.2f} seconds")
        print(f"Drowsy state: {drowsy_state}")

        # Drowsy State Branch Logic
        if drowsy_state is True:
            cv2.rectangle(frame, (400, 20), (512, 60), (255, 255, 255), -1)
            cv2.putText(frame, 'Drowsy', (400, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            #record the first time drowsiness detected by multiplying current frame number with frame duration
            if prev_drowsy_state is False:
                play_audio_thread = threading.Thread(target=play_audio, args=(warning_sound,))
                play_audio_thread.start()
            prev_drowsy_state=True
        else:
            prev_drowsy_state = False

        """
        Static Information Display Function
        """
        # drawing and writing the annotation
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
