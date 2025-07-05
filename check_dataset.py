import os

train_dir = "C:/Users/rakes/Desktop/face emotion detection/facial_emotion_recognition/dataset/train" #A ADAPTER


emotion_classes = os.listdir(train_dir) #on list


for emotion in emotion_classes: #count the number of images
    path = os.path.join(train_dir, emotion)
    num_images = len(os.listdir(path))
    print(f"{emotion}: {num_images} images")
