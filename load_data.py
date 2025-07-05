import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'C:/Users/rakes/Desktop/face emotion detection/facial_emotion_recognition/dataset/train' #A ADAPTER
test_dir = 'C:/Users/rakes/Desktop/face emotion detection/facial_emotion_recognition/dataset/test' #A ADAPTER


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  
    color_mode='grayscale',  # Load images in grayscale
    batch_size=32,
    class_mode='categorical')  # Categorical for one-hot labels



test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),  # Resize images to 48x48
    color_mode='grayscale',  # Load images in grayscale
    batch_size=32,
    class_mode='categorical') 


__all__ = ['train_generator', 'test_generator'] # Accessibility
