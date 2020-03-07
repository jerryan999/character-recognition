# -*- encoding: utf-8
import os
import sys
import numpy as np
import cv2


def load_img(fn):
    img = cv2.imread(fn)
    return img

def save_img(img, fn):
    status = cv2.imwrite(fn, img)
    if status is False:
        cv2.imshow(fn, img)
        cv2.waitKey(0)

def split_image_into_subimage(image, width_max=40, width_min=10):
    # 将每张图片切割成四个字符，用于后续训练
    # width_max: 限制最大宽度
    # width_min: 限制最小宽度
    # min_area 限制最小面积
    
    # 外围加一个padding
    image_with_bord = cv2.copyMakeBorder(image,1,1,1,1, cv2.BORDER_CONSTANT,value=[255]) 
    
    # 找轮廓算法
    # img, contours, hierarchy = cv2.findContours(image_with_bord.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(image_with_bord.copy() ,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    sub_images = []
    sub_images_x = []  
    for cnt in contours:
        # 最小的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        if x != 0 and y != 0 and w>width_min:   #w*h >= min_area:
            # 显示图片
            # print(w)
            if w > width_max:
                w = int(w/2)
                sub_images.append(image_with_bord[y:y+h, x:x+w])
                sub_images_x.append(x)
                
                sub_images.append(image_with_bord[y:y+h, x+w:x+2*w])
                sub_images_x.append(x)
                continue
            sub_images.append(image_with_bord[y:y+h, x:x+w])
            sub_images_x.append(x)
            
    seq_index = np.argsort(sub_images_x)
    return [sub_images[i]  for i in seq_index]


def resize_with_padding(image,desired_height=40,desired_width=40):
    # The main idea is to first resize the input image so that its maximum size 
    # equals to the given size. Then we pad the resized image to make it square. 
    # A number of packages in Python can easily achieves this
    old_size = image.shape[:2]   # old_size is in (height, width) format
   
    ratio = float(max([desired_height,desired_width]))/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    im = cv2.resize(image, (new_size[1], new_size[0]))
    
    delta_w = desired_width - new_size[1]
    delta_h = desired_height - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

def gen_sub_img(raw_img,sub_img_num=4):
    # generate $sub_img_num$ images from raw captcha image through a series of processes
    # 
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.medianBlur(gray_img, 3)
    ret,binary_img = cv2.threshold(blur_img,200, 255, cv2.THRESH_BINARY)

    # split according to min_length * min_width
    sub_imgs = split_image_into_subimage(binary_img)
    if len(sub_imgs)==sub_img_num:
        for s in sub_imgs:
            s_with_padding = resize_with_padding(s,desired_height=40,desired_width=40)
            yield s_with_padding

def split_img(src_folder):
    # debug purpose to split all src_folder's captcha image
    img_list = [i for i in os.listdir(src_folder) if i.endswith('jpg')]
    for index, img_name in enumerate(img_list):
        raw_img = load_img(os.path.join(src_folder, img_name))
        for sub_index, sub_img in enumerate(gen_sub_img(raw_img)):
            cv2.imshow(img_name[sub_index], sub_img)
            cv2.waitKey(0)
