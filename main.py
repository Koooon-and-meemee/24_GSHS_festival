import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
from keras.models import load_model
import pygame

prev_predection = 0

def find_h5_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    return h5_files

def preprocess_img(img):
    # 이미지 크기 조정
    img = cv2.resize(img, (32, 32))
    # 이미지 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지 정규화
    gray = gray / 255.0
    # 모델 입력 형태에 맞게 차원 추가
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=-1)
    return gray

def camera_logic(model_addr):

  model = load_model('model_addr')
  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  while True:
      ret, frame = capture.read()
      preprocessed_img = preprocess_img(frame)

      # 예측
      prediction = model.predict(preprocessed_img)

      # 첫 번째 인덱스 확률이 가장 높으면 1 출력, 아니면 0 출력
      prediction = np.argmax(prediction[0])

      if prev_predection == 1 and prediction == 0:
          time.sleep(1)
          continue
      
      if prediction == 0 and prediction == 1:
          prev_prediction = prediction
          time.sleep(1)
          continue

      time.sleep(1)

      if cv2.waitKey(33) == ord('q'):
        break


  return



current_directory = "c:/Users/parksangwon/Documents/" ##파일 찾을 위치

h5_files_in_current_directory = find_h5_files(current_directory)

print("\n")
print("List of currently owned models :")

for file_path in h5_files_in_current_directory:
  print(file_path)

n = input("Please enter the number of the moedl you want to run :")

model_add = h5_files_in_current_directory[n]

while True:
    ans = input("Press S to start :")
    if ans == "S":
        break

camera_logic(model_add)

capture.release()
