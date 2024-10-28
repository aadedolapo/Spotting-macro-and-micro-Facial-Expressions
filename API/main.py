import numpy as np
import pandas as pd
import csv


# visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px

import cv2
import torch
import torch.nn.functional as F


# API Libraries
from fastapi import FastAPI, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
import aiofiles
import asyncio
import os

app = FastAPI()


model = torch.hub.load('pytorch/vision:v0.10.0',
                       'inception_v3', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
state_dict = torch.load(
    r"C:\Users\kings\OneDrive - MMU\MSC DATA SCIENCE\MSC Project\models\Inception3_Adamproposed_augmediapipe.pt")

model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


@app.get('/predict')
def predict_emotion(file: UploadFile):

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    temp = NamedTemporaryFile(delete=False)
    cap = cv2.VideoCapture(temp.name)
    if not cap.isOpened():
        # Pass temp.name to VideoCapture()
        raise IOError("Cannot open webcam")

    result_data = []

    while True:
        # Capture a frame
        ret, frame = cap.read()
        facedet = cv2.CascadeClassifier(
            '\haarcascades\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedet.detectMultiScale(gray_frame, 1.1, 4)

        if len(faces) == 0:
            print("No faces detected in the frame")
            continue  # Skip processing this frame if no faces are detected

        for x, y, w, h in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            facess = facedet.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("Face not detected")
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
        reshaped_data = reshaped_data.unsqueeze(0)  # add a fourth dimension
        outputs = model(reshaped_data)
        pred = F.softmax(outputs[0], dim=-1)
        final_pred = torch.argmax(pred, 0)

        if final_pred == 0:
            emotion = "Negative"
        elif final_pred == 1:
            emotion = "Neutral"
        else:
            emotion = "Positive"

        text_x, text_y = x, y - 10  # Adjust the position above the bounding box
        cv2.putText(frame, emotion, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        conf, classes = torch.max(pred, -1)
        emotion_id = [0, 1, 2]
        class_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        class_names = [class_dict[label] for label in emotion_id]
        result_data.append({'Confidence': conf.item(
        ),  'Emotion': class_names[classes.item()]})

        # Write the result to a CSV file
        with open('result.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                # Write header only if the file is empty
                writer.writerow(['Confidence', 'Emotion'])
            writer.writerows(result_data)

        cv2.imshow("Facial Expression Recognition", frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.read_csv('result.csv')
    return {"predictions": df.to_dict()}
