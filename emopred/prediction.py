import cv2
import csv
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

model = torch.hub.load('pytorch/vision:v0.10.0',
                       'inception_v3', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
state_dict = torch.load("models\Inception3_Adamproposed_augmediapipe.pt")

model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


def create_prediction_video(frames, storage, timestamp, model_type, fps=4, is_color=True):
    """Creates a video from a list of frames and saves it."""
    video_path = storage / f"pred_{model_type}_{timestamp}.mp4"
    height, width = frames[0].shape[:2]

    # Ensure frames are in uint8 format
    frames_uint8 = [(frame * 255).astype(np.uint8) if frame.max()
                    <= 1 else frame.astype(np.uint8) for frame in frames]

    # Initialize VideoWriter
    out = cv2.VideoWriter(str(video_path),
                          cv2.VideoWriter_fourcc(*'avc1'),
                          fps, (width, height),
                          is_color)

    if not out.isOpened():
        out = cv2.VideoWriter(str(video_path),
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height),
                              is_color)

    for frame in frames_uint8:
        out.write(frame)

    out.release()
    print(f"Saved video at: {video_path}")
    return str(video_path)


def predict_emotion(video_path, output_folder):
    """Processes a video, detects emotions, and saves results + processed video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    facedet = cv2.CascadeClassifier(
        r'C:\Users\kings\OneDrive - MMU\MSC DATA SCIENCE\MSC Project\Msc-Project\haarcascades\haarcascade_frontalface_default.xml')

    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedet.detectMultiScale(gray_frame, 1.1, 4)

        if len(faces) == 0:
            print("No faces detected in the frame")
            continue

        for x, y, w, h in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        rgb_image = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(rgb_image, (299, 299))
        image = image / 255.0  # nomalization
        data = torch.from_numpy(image).type(torch.FloatTensor).to(device)
        # Reshape the tensor to have 3 as the first dimension and adjust dimensions for model
        reshaped_data = data.permute(2, 0, 1).unsqueeze(0)

        outputs = model(reshaped_data)
        pred = F.softmax(outputs[0], dim=-1)
        final_pred = torch.argmax(pred, 0)

        emotion = ["Negative", "Neutral", "Positive"][final_pred.item()]
        text_x, text_y = x, y - 10
        cv2.putText(frame, emotion, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Confidence and emotion label
        conf, classes = torch.max(pred, -1)
        result_data = [[conf.item(), pred.tolist(), emotion]]

        # Save results to CSV
        csv_path = Path(output_folder) / \
            (Path(video_path).stem + '_results.csv')
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Confidence', 'max_confidence', 'Emotion'])
            writer.writerows(result_data)

        cv2.imshow("Facial Expression Recognition", frame)
        processed_frames.append(frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save processed frames as a video
    timestamp = Path(video_path).stem
    storage_path = Path(output_folder)
    saved_video_path = create_prediction_video(
        processed_frames, storage_path, timestamp, "emotion_detection", fps=4, is_color=True)

    print(f"Video saved at: {saved_video_path}")


if __name__ == '__main__':
    predict_emotion()
