
import cv2 # type: ignore
from keras.models import load_model # type: ignore
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochCategorica.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

image = cv2.imread('C:\\Brain Tumor Image Classification\\pred\\pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)
input_img=np.expand_dims(img, axis=0)

result=model.predict_classes(input_img)

print(result)
