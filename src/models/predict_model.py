from keras.models import model_from_json
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]

APPEARED_LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]

CATEGORY_TO_CAPTCHA = dict(zip(range(len(APPEARED_LETTERS)), APPEARED_LETTERS))

with open('models/model_num.json', 'r') as json_file:
    model = model_from_json(json_file.read())
    model.load_weights("models/model_num-1583664747.61.h5")

img_names = ['P/P:1048.jpg','P/P:2388.jpg','H/H:1030.jpg','E/E:1168.jpg','6/6:108.jpg','Z/Z:1095.jpg']
for img_name in img_names:
	img = cv2.imread("data/processed/{}".format(img_name),cv2.IMREAD_UNCHANGED)
	x = img.reshape(1,40,40,1)
	predict_category = model.predict_classes(x)
	print("image named {} predicted class is:{}".format(img_name,CATEGORY_TO_CAPTCHA[predict_category[0]]))