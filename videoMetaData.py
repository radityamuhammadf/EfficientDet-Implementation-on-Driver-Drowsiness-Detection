import os

current_directory = os.getcwd()
videos_metadata = {

    ### Debugging Purpose

    # 'debug_webcam_input': {
    #     'path': 0,
    #     'light_sufficient': True,
    #     'looking_lr': False,
    #     'detected_drowsiness': [],
    #     'ground_truth_drowsiness': [],
    #     'detection_accuracy':0,
    #     'inference_time':0,
    #     'profiler_result':""
    # },
    # 'debug_video_input': {
    #     'path': os.path.join(current_directory, r'test_video/debugging_sample.avi'),
    #     'light_sufficient': True,
    #     'looking_lr': False,
    #     'detected_drowsiness': [],
    #     'ground_truth_drowsiness': [],
    #     'detection_accuracy':0,
    #     'inference_time':0,
    #     'profiler_result':""
    # },

    ### False Posiitve Test

    'false_positive-test': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/false_positive-test.mp4'),
        'light_sufficient': True,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },

    ## With Glasses Test

    'glasses-afternoon-looking_lr': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/glasses-afternoon-looking_lr.mp4'),
        'light_sufficient': False,
        'looking_lr': True,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,22,26,39,44,47,56],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'glasses-afternoon-straight_fw': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/glasses-afternoon-straight_fw.mp4'),
        'light_sufficient': False,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [17,22,24,39,43,46,58],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'glasses-night-looking_lr': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/glasses-night-looking_lr.mp4'),
        'light_sufficient': False,
        'looking_lr': True,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,25,39,45,46,61],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'glasses-night-straight_fw': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/glasses-night-straight_fw.mp4'),
        'light_sufficient': False,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,22,24,38,45,47,58],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'glasses-noon-looking_lr': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/glasses-noon-looking_lr.mp4'),
        'light_sufficient': True,
        'looking_lr': True,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [15,24,34,43,55],
        'detection_accuracy':0,
        'inference_time':0,
        'CPU':0,
        'GPU':0,
        'RAM':0
    },
    'glasses-noon-straight_fw': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/glasses-noon-straight_fw.mp4'),
        'light_sufficient': True,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [14,24,34,43,54],
        'detection_accuracy':0,
        'inference_time':0,
        'CPU':0,
        'GPU':0,
        'RAM':0
    },

    ## No Glasses Test

    'no_glasses-afternoon-looking_lr': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/no_glasses-afternoon-looking_lr.mp4'),
        'light_sufficient': False,
        'looking_lr': True,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [17,23,25,39,44,45,58],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'no_glasses-afternoon-straight_fw': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/no_glasses-afternoon-straight_fw.mp4'),
        'light_sufficient': False,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [17,24,25,38,44,46,56],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'no_glasses-night-looking_lr': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/no_glasses-night-looking_lr.mp4'),
        'light_sufficient': False,
        'looking_lr': True,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,24,26,39,46,47,61],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'no_glasses-night-straight_fw': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/no_glasses-night-straight_fw.mp4'),
        'light_sufficient': False,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,24,26,38,44,47,59],
        'detection_accuracy':0,
        'inference_time':0,
        'profiler_result':""
    },
    'no_glasses-noon-looking_lr': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/no_glasses-noon-looking_lr.mp4'),
        'light_sufficient': True,
        'looking_lr': True,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,22,24,39,44,47,59],
        'detection_accuracy':0,
        'inference_time':0,
        'CPU':0,
        'GPU':0,
        'RAM':0
    },
    'no_glasses-noon-straight_fw': {
        'path': os.path.join(current_directory, r'Research_DDD_VideoEvaluation/no_glasses-noon-straight_fw.mp4'),
        'light_sufficient': True,
        'looking_lr': False,
        'detected_drowsiness': [],
        'ground_truth_drowsiness': [18,23,26,37,43,46,59],
        'detection_accuracy':0,
        'inference_time':0,
        'CPU':0,
        'GPU':0,
        'RAM':0
    }
}