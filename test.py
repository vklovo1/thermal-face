import os

import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

interpreter = tflite.Interpreter(model_path="thermal_face_automl_edge_fast.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_path = "mask.jpg"

og_img = cv2.imread(img_path)
img = cv2.resize(og_img, (192, 192))

test_image = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])

interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()

bbox = interpreter.get_tensor(output_details[0]['index'])
nesto = interpreter.get_tensor(output_details[2]['index'])
print(nesto)
print(bbox)

y_min, x_min, y_max, x_max = bbox[0][0] * 192

im = np.array(Image.open('mask.jpg'), dtype=np.uint8)

fig, ax = plt.subplots(1)

ax.imshow(im)

rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor='black', facecolor='none')

ax.add_patch(rect)
plt.imshow(im)
plt.show()
