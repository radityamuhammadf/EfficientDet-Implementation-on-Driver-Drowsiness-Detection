import cv2
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
from torchvision.transforms import transforms

# Load the model
def load_model(model_path, device):
    config = get_efficientdet_config('tf_efficientdet_d1')
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    model = DetBenchPredict(net)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocess the input frame
def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# Postprocess the output to extract bounding boxes and labels
def postprocess_detections(detections, threshold=0.4):
    results = []
    for det in detections:
        boxes, scores, labels = det['boxes'], det['scores'], det['labels']
        for box, score, label in zip(boxes, scores, labels):
            if score > threshold:
                results.append((box, score, label))
    return results

# Main function to run object detection on webcam
def main():
    model_path = r'D:\01-KULIAH\0-SEMESTER 8\01-Undergraduate Thesis\0-Laboratory\EfficientDet Inference\Yet-Another-EfficientDet-Pytorch\logs\blink_yawn\efficientdet-d0_17_11400.pth'  # replace with your model path
    device = torch.device('cpu')
    
    model = load_model(model_path, device)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        input_tensor = preprocess_image(frame, device)
        
        # Run inference
        with torch.no_grad():
            detections = model(input_tensor)
        
        # Postprocess detections
        results = postprocess_detections(detections)
        
        # Draw results on the frame
        for box, score, label in results:
            box = box.int().tolist()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'Label: {label}, Score: {score:.2f}', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('EfficientDet Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
