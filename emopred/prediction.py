import cv2
import torch
import torch.nn.functional as F
import uuid
import os


print(os.getcwd())

model = torch.hub.load('pytorch/vision:v0.10.0',
                       'inception_v3', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
state_dict = torch.load("models\Inception3_Adamproposed_augmediapipe.pt")

model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


def predict_emotion(input_video):
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Default to 20 if FPS is not available
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_video_name = f"output_{uuid.uuid4()}.mp4"
    out = cv2.VideoWriter(output_video_name, fourcc,
                          frame_rate, (frame_width, frame_height))

    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frame is captured

        facedet = cv2.CascadeClassifier(
            'haarcascades\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedet.detectMultiScale(gray_frame, 1.1, 4)

        if len(faces) == 0:
            continue  # Skip processing this frame if no faces are detected

        for x, y, w, h in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            facess = facedet.detectMultiScale(roi_gray)
            if len(facess) == 0:
                continue
            else:
                for (ex, ey, ew, eh) in facess:
                    # cropping the face
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]

        rgb_image = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(rgb_image, (299, 299))
        image = image / 255.0  # normalization
        data = torch.from_numpy(image)
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        # Reshape the tensor to have 3 as the first dimension
        reshaped_data = data.permute(2, 0, 1)
        reshaped_data = reshaped_data.unsqueeze(0)
        outputs = model(reshaped_data)
        pred = F.softmax(outputs[0], dim=-1)
        final_pred = torch.argmax(pred, 0)

        if final_pred == 0:
            emotion = "Negative"
        elif final_pred == 1:
            emotion = "Neutral"
        else:
            emotion = "Positive"

        text_x, text_y = x, y - 10
        cv2.putText(frame, emotion, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        out.write(frame)
    out.release()


if __name__ == '__main__':
    predict_emotion(0)
