import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from torch.cuda.amp import autocast
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
import os
import torch
import openpyxl
from openpyxl.styles import Font, Alignment
from jtop.jtop import jtop
from datetime import datetime


def main():
    test_time_start=time.time()
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
    cudnn.benchmark = False
    obj_list = ['closed-eyes', 'yawn']

    # Initialize the dictionary to keep track of detection times and last durations
    detections = {
        'closed-eyes': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None},
        'yawn': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None}
    }
    # # Initialize so-called tracking logic dictionary -> for recording latest detection if there's no detection
    # prev_annotation={
    #     'inference_results':[],
    #     'rep_count':12 # -> delaying maximum 12 frames if the current frame don't have any detection result (12 frames equal to 0.36 [by 12*0.03])
    # }

    #Drowsy State Declaration
    prev_drowsy_state=False
    drowsy_state = False
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model_path = os.path.join(current_directory,r"saved_weights/efficientdet-d0_19_12760.pth")
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


    # Excel Export Workbook Initiation
    # Create Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Videos Metadata"
    # Create Header Row
    # Write the header row
    headers = [
        "Name", "Light", "Look", 
        "Predict","G. Truth","Accuracy (%)", "FP Rate (%)", 
        "Inf. Time (ms)", "CPU (%)", "GPU (%)","RAM (MB)"
    ]
    ws.append(headers)
    # Set column width and make header text bold
    for col in ws.iter_cols(min_row=1, max_row=1, min_col=1, max_col=len(headers)):
        for cell in col:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')
            ws.column_dimensions[cell.column_letter].width = 14
    ws.column_dimensions['A'].width = 25

    #creating videos_metadata dictionary
    videos_metadata = {
        # 'webcam_input': {
        #     'path': 0,
        #     'light_sufficient': True,
        #     'looking_lr': False,
        #     'detected_drowsiness': [],
        #     'ground_truth_drowsiness': [],
        #     'detection_accuracy':0,
        #     'false_positive_rate':0,
        #     'inference_time':0,
        #     'profiler_result':""
        # },
        # 'debug_video_sample': {
        #     'path': os.path.join(current_directory, r'test_video/debugging_sample.avi'),
        #     'light_sufficient': True,
        #     'looking_lr': False,
        #     'detected_drowsiness': [],
        #     'ground_truth_drowsiness': [],
        #     'detection_accuracy':0,
        #     'false_positive_rate':0,
        #     'inference_time':0,
        #     'profiler_result':""
        # },
        'light_sufficient-glasses': {
            'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/light_sufficient-glasses.mp4'),
            'light_sufficient': True,
            'looking_lr': False,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [14,24,34,43,54],
            'detection_accuracy':0,
            'false_positive_rate':0,
            'inference_time':0,
            'CPU':0,
            'GPU':0,
            'RAM':0
        },
        'light_sufficient-looking_lr-glasses': {
            'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/light_sufficient-looking_lr-glasses.mp4'),
            'light_sufficient': True,
            'looking_lr': True,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [15,24,34,43,55],
            'detection_accuracy':0,
            'false_positive_rate':0,
            'inference_time':0,
            'CPU':0,
            'GPU':0,
            'RAM':0
        },
        'low_light-glasses': {
            'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/low_light-glasses.mp4'),
            'light_sufficient': False,
            'looking_lr': False,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [14,25,34,45,56],
            'detection_accuracy':0,
            'false_positive_rate':0,
            'inference_time':0,
            'CPU':0,
            'GPU':0,
            'RAM':0
        },
        'low_light-looking_lr-glasses': {
            'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/low_light-looking_lr-glasses.mp4'),
            'light_sufficient': False,
            'looking_lr': True,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [14,23,36,44,56],
            'detection_accuracy':0,
            'false_positive_rate':0,
            'inference_time':0,
            'CPU':0,
            'GPU':0,
            'RAM':0
        },
        # 'low_light-looking_lr': {
        #     'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/low_light-looking_lr.mp4'),
        #     'light_sufficient': False,
        #     'looking_lr': True,
        #     'detected_drowsiness': [],
        #     'ground_truth_drowsiness': [],
        #     'detection_accuracy':0,
        #     'false_positive_rate':0,
        #     'inference_time':0,
        #     'CPU':0,
        #     'GPU':0,
        #     'RAM':0
        # },
        # 'low_light': {
        #     'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/low_light.mp4'),
        #     'light_sufficient': False,
        #     'looking_lr': False,
        #     'detected_drowsiness': [],
        #     'ground_truth_drowsiness': [16,25,35,43,56],
        #     'detection_accuracy':0,
        #     'false_positive_rate':0,
        #     'inference_time':0,
        #     'CPU':0,
        #     'GPU':0,
        #     'RAM':0
        # },

    }

    #iterating every metadata element in every 'video_name' (videos_metadata members) element
    for video_name, metadata in videos_metadata.items():
        video_path = metadata['path']
        # Start the webcam
        cap = cv2.VideoCapture(video_path)
        # cap = cv2.VideoCapture(os.path.join(current_directory, r"test_video\10-MaleGlasses-Trim.avi"))  
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1 / fps
        frame_number = 0
        print("FPS: ", fps)
        temp_inference_time = []

        temp_resource={
            'CPU':[],
            'GPU':[],
            'RAM':[]
        }

        temp_detected_drowsiness = []


        try:
            with jtop() as jetson:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_number += 1
                    # Resize the frame to 512x384 -> 4:3 aspect ratio for 512 input size
                    frame = cv2.resize(frame, (512, 384)) 

                    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

                    if use_cuda:
                        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
                    else:
                        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

                    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

                    with torch.no_grad():
                        with autocast():
                            inference_start=time.time()
                            features, regression, classification, anchors = model(x)
                            inference_end=time.time()
                            temp_inference_time.append(inference_end-inference_start)
                            # Profiling Jetson Stats
                            temp_resource['CPU'].append((jetson.stats['CPU1']+jetson.stats['CPU2']+jetson.stats['CPU3']+jetson.stats['CPU4']+jetson.stats['CPU5']+jetson.stats['CPU6'])/6)
                            temp_resource['GPU'].append(jetson.stats['GPU'])
                            temp_resource['RAM'].append(jetson.stats['RAM'])


                    out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                                      iou_threshold)

                    out = invert_affine(framed_metas, out)

                    # # Check if any detections were made
                    # if not any(len(detection['rois']) > 0 for detection in out):
                    #     if prev_annotation['rep_count'] > 0 and prev_annotation['inference_results']:
                    #         features, regression, classification, anchors = prev_annotation['inference_results']
                    #         out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                    #                         iou_threshold)
                    #         out = invert_affine(framed_metas, out)
                    #         print("DEBUG- using previously detected annotation")
                    #         prev_annotation['rep_count'] -= 1
                    # else: #check if there's detection been made
                    #     prev_annotation['inference_results'] = [features, regression, classification, anchors]
                    #     prev_annotation['rep_count'] = 12

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
                    if closed_eyes_duration > 0.5 or yawn_duration > 5.0:  # thresholds in seconds
                        drowsy_state = True
                    else:
                        drowsy_state = False

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
                            temp_detected_drowsiness.append(frame_number*frame_duration)
                            metadata['detected_drowsiness'].append(frame_number*frame_duration)                            
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
                    inferece_time_avg=(sum(temp_inference_time) / len(temp_inference_time))*1000

            cap.release()    

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Profiling Jetson Stats               
            print("\nDEBUGGING STATS\n")
            # Profiling Result
            gpu_usage = sum(temp_resource['GPU'])/len(temp_resource['GPU'])
            # Collecting all CPU usage and calculate the average
            cpu_usage = sum(temp_resource['CPU'])/len(temp_resource['CPU'])
            memory_usage = sum(temp_resource['RAM'])/len(temp_resource['RAM'])

            # Append Profiling Result to Metadata
            metadata['CPU']=f"{cpu_usage:.2f}"
            metadata['GPU']=f"{gpu_usage:.2f}"
            metadata['RAM']=f"{memory_usage:.2f}"
            metadata['inference_time'] = f"{inferece_time_avg:.2f}"
            metadata['detected_drowsiness']=f"{metadata['detected_drowsiness']}"
            metadata['ground_truth_drowsiness']=f"{metadata['ground_truth_drowsiness']}"

            # Debugging Prompt
            print(f"CPU Usage: {metadata['CPU']}%")
            print(f"GPU Usage: {metadata['GPU']}%")
            print(f"Memory Usage: {metadata['RAM']}MB")

            # Measure and print other metadata
            # Detection Accuracy Measurements
            if len(metadata['ground_truth_drowsiness']) != 0:
                if len(temp_detected_drowsiness)<=len(metadata['ground_truth_drowsiness']): #stating if there's no false positive by make sure the ground truth are more than equal to detected state 
                    metadata['detection_accuracy']=len(temp_detected_drowsiness)/len(metadata['ground_truth_drowsiness'])
                    metadata['false_positive_rate']=0
                else: #for if the detected state exceed the ground truth 
                    # find exceeding values by subsracting detected drowsiness with ground truth value
                    exceed_value=len(temp_detected_drowsiness)-len(metadata['ground_truth_drowsiness'])
                    metadata['detection_accuracy']=(len(metadata['ground_truth_drowsiness'])-exceed_value)/len(metadata['ground_truth_drowsiness'])
                    metadata['false_positive_rate']=exceed_value/len(metadata['ground_truth_drowsiness'])
            print(f"Average Inference Time: {metadata['inference_time']}ms") # debugging prompt
            
            #Append Row Using Video Metadata Information
            row = [
                video_name,
                metadata['light_sufficient'],
                metadata['looking_lr'],
                metadata['detected_drowsiness'],
                metadata['ground_truth_drowsiness'],
                metadata['detection_accuracy'],
                metadata['false_positive_rate'],
                metadata['inference_time'],
                metadata['CPU'],
                metadata['GPU'],
                metadata['RAM']
            ]
            ws.append(row)
    
    #Append How Long The Test is Going
    test_time_end=time.time()
    test_dur_min,test_dur_sec=divmod((test_time_end-test_time_start),60)
    ws.append([f"This test took {int(test_dur_min)} minutes and {test_dur_sec:.2f} seconds"])
    print(f"This test took {int(test_dur_min)} minutes and {test_dur_sec:.2f} seconds")
    
    # Export Excel File
    # Generate filename
    now = datetime.now()
    date_str = now.strftime("%b%d-%H%M")
    dynamic_filename = f"EfficientDetD0-MainTest-{date_str}.xlsx" #For Debugging
    # dynamic_filename = f"VideoTest-Main-{date_str}.xlsx" #For The Main Test Set
    output_file=os.path.join(current_directory, f"video-test_result/{dynamic_filename}")
    wb.save(output_file)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
