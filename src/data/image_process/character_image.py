import os
import cv2
import numpy as np
import os
from image_process import preprocess
import logging
from captcha.constant import CAPTCHA_TO_CATEGORY,CATEGORY_TO_CAPTCHA,APPEARED_LETTERS 

logger = logging.getLogger(__name__)

def load_data(folder):
	# prepare X and y for processing
	img_list = [i for i in os.listdir(folder) if i.endswith('jpg')]

	count = 0
	for img_index,img_name in enumerate(img_list):
		img_path = os.path.join(folder,img_name)
		img = preprocess.load_img(img_path)
		sub_imgs = preprocess.gen_sub_img(raw_img=img,sub_img_num=4)
		for sub_index, sub_img in enumerate(sub_imgs):
			sub_img_shaped = sub_img.reshape(1,40,40,1)
			count += 1
			if count == 1:
				# data = np.ndarray((1,40,40,1)) 
				# label = np.ndarray(1,)
				data = sub_img_shaped.copy()
				label = np.array([CAPTCHA_TO_CATEGORY[img_name[sub_index]]])
			else:
				data = np.vstack((data,sub_img_shaped))
				label = np.append(label,[CAPTCHA_TO_CATEGORY[img_name[sub_index]]],axis=0)

			if count % 100 ==0:
				logger.info("{} letters of captcha loaded".format(count))

	return data, label

def output_character_images(input_folder,output_folder):
	# 验证码图片切割成一个一个的character，并存在文件夹中
	d, l = load_data(input_folder)

	# 新建文件夹
	for ch in APPEARED_LETTERS:
		folder = '{}/{}'.format(output_folder,ch)
		if not os.path.exists(folder):
			os.mkdir(folder)

	# 写入到目标文件夹
	for n, i in enumerate(d):
		char = CATEGORY_TO_CAPTCHA[l[n]]
		cv2.imwrite('{}/{}/{}:{}.jpg'.format(output_folder,char, char, n),i)

if __name__ == '__main__':
	d, l = load_data('./samples')

	import os
	for ch in APPEARED_LETTERS:
		folder = 'chars/{}'.format(ch)
		if not os.path.exists(folder):
			os.mkdir(folder)

	for n, i in enumerate(d):
		char = CATEGORY_TO_CAPTCHA[l[n]]
		cv2.imwrite('chars/{}/{}:{}.jpg'.format(char, char, n),i)