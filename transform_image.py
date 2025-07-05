import cv2
import os


def transform_image(image_path, output_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (48, 48)) #size
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) #grayish tone
    
    
    cv2.imwrite(output_path, img_gray) #registration
    print(f"Image transformed and saved to {output_path}")


input_image_path = 'C:/Users/rakes/Pictures/Training_2913.jpg' #A ADAPTER
output_image_path = 'C:/Users/rakes/Pictures/Training_2913_transformed.jpg' #A ADAPTER

transform_image(input_image_path, output_image_path)
