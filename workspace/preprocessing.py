"""
This file contains a helper function for data preprocessing
"""
import os
import cv2

import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def get_images_path(annot_path):
# Use os.path.basename to get the file name

  img_path = annot_path.split(os.path.sep)
  img_path.pop(-2)
  img_path[-1] = img_path[-1][:-4] + '.jpg'
  img_path[0]  = '/' + img_path[0]
  img_path = os.path.join(*img_path)
  # Now 'file_name' contains the name of the file
  return img_path

def images_labels_pathes(categories_dirs: list):
  """
  This function returns two lists one for annotations pathes 
  and another for images pathes
  """

  annot_pathes = [
                        # getting the pathes of .xml file
                        os.path.join(categorie_dir,'Label',file_name)
                        for categorie_dir in categories_dirs
                        for file_name in os.listdir(categorie_dir + '/Label')
                        # ensuring  the file extension is xml
                        if file_name.endswith('.txt')
                        ]
  
  images_pathes = [
                            get_images_path(annot_path)
                            for annot_path in annot_pathes
                        ]
  return annot_pathes, images_pathes



def load_image(image_path):
  image = load_img(image_path)
  image = img_to_array(image)
  return image

def load_labels(annot_pathes, classes_map):
  all_boxes   = []
  all_classes = []
  for label_path in annot_pathes:
    labels = np.loadtxt(label_path,dtype='str')
    classes = []
    boxes   = []
    if len(labels.shape) == 1:
      label = labels.tolist()
      if len(label) == 6:
        label[0] = label[0] + ' ' + label[1]
        label.pop(1)
      if len(label) == 7:
        label[0] = label[0] + ' ' + label[1] + ' ' + label[2]
        label.pop(1)
        label.pop(1)

      classes.append(label[0])
      boxes.append(label[1:])
    else:
      for label in labels:
        label = label.tolist()
        if len(label) == 6:
          label[0] = label[0] + ' ' + label[1]
          label.pop(1)
        if len(label) == 7:
          label[0] = label[0] + ' ' + label[1] + ' ' + label[2]
          label.pop(1)
          label.pop(1)

        classes.append(label[0])
        boxes.append(label[1:])
    # convert classes to int
    classes = [
                    classes_map[cls]
                    for cls in classes
    ]

    # convert boxes to float
    boxes = [list(map(float,box)) for box in boxes]

    all_boxes.append(boxes)
    all_classes.append(classes)
  return all_boxes, all_classes





def transform_xyxy_xywh(bboxes: list):
  bboxes = np.array(bboxes)
  x_min, y_min, x_max, y_max = np.split(bboxes, 4, axis=1)
  width = x_max - x_min
  height = y_max - y_min
  x = (x_max + x_min)/2
  y = (y_max + y_min)/2
  return np.concatenate([x, y, width, height], axis=1).tolist()
  


def scale_bbox(bbox, image, o_shape):
    W, H, C = o_shape
    w, h, c = image.shape
    Wratio = w/W
    Hratio = h/H
    ratio = np.array([Hratio, Wratio, Hratio, Wratio])
    bbox = bbox * ratio
    return bbox, ratio

def scale_with_padding(image, bboxs, base=512):
    """
    Scale image and bboxes with saving aspect ratio for the image
    """
    image = load_image(image)
    H, W, C = image.shape
    bboxs = np.array(bboxs)
    if H > W:
        height_persentage = float(base/H)
        width_size = int(W*height_persentage)
        n_shape = (width_size,base)
        resized_image = cv2.resize(image, n_shape, interpolation=cv2.INTER_CUBIC)
        h, w, c = resized_image.shape
        bbox, ratio = scale_bbox(bboxs, resized_image, (H, W, C))
        width1 = (base - w) // 2
        width2 = (base - w) - width1
        bbox = np.concatenate((bbox[:,0:1]+width1, bbox[:,1:2], bbox[:,2:3]+width2, bbox[:,3:]),-1)
        
        # Symmetric Padding
        mask = np.array(np.zeros(shape=(base, width1, C)), dtype=int)
        resized_image = np.concatenate((resized_image, mask), axis=1)

        mask = np.array(np.zeros(shape=(base, width2, C)), dtype=int)
        resized_image = np.concatenate((mask, resized_image), axis=1)
        # display(resized_image, bbox)
        
    else:
        width_percentage = float(base/W)
        height_size = int(H*width_percentage)
        n_shape = (base, height_size)
        resized_image = cv2.resize(image, n_shape, interpolation=cv2.INTER_CUBIC)
        h, w, c = resized_image.shape
        bbox, ratio = scale_bbox(bboxs, resized_image, (H, W, C))
        height1 = (base - h) // 2
        height2 = (base - h) - height1
        bbox = np.concatenate((bbox[:,0:1], bbox[:,1:2]+height1, bbox[:,2:3], bbox[:,3:]+height2),-1)
        
        # Symmetric Padding
        mask = np.array(np.zeros(shape=(height1, base, C)), dtype=int)
        resized_image = np.concatenate((resized_image, mask))
        
        mask = np.array(np.zeros(shape=(height2, base, C)), dtype=int)
        resized_image = np.concatenate((mask, resized_image))

    return bbox.tolist(), n_shape, ratio

def analize_data(images_pathes, classes_list, sorting = False):
  analysis_dict  = {}
  for class_ in classes_list:
    num_imgs = [1
                for images_pathes in images_pathes
                if images_pathes.split(os.path.sep)[-2] == class_]
    num_imgs = len(num_imgs)
    analysis_dict[class_] = num_imgs
  if sorting:
    sorted_analysis_dict = {k: v for k, v in sorted(analysis_dict.items(), key=lambda item: item[1])}
    return sorted_analysis_dict
  return analysis_dict

def visualize_analysis(analysis_dict, title="", figsize = (15,5), plot_nums = True):
  plt.figure(figsize=figsize)
  # Create a barplot using seaborn
  sns.barplot(x=list(analysis_dict.keys()), y=list(analysis_dict.values()), color='skyblue', edgecolor='black')

  # Adding labels and title
  plt.xlabel('Class', size = 15)
  plt.ylabel('Count', size = 15)
  plt.title(title, size = 25)
  if plot_nums:
    for i, count in enumerate(list(analysis_dict.values())):
      plt.annotate(str(count), xy=(i, count), ha='center', va='bottom')
      
def get_images_path(annot_path):
  # Use os.path.basename to get the file name
  img_path = annot_path.split(os.path.sep)
  img_path.pop(-2)
  img_path[-1] = img_path[-1][:-4] + '.jpg'
  img_path[0]  = '/' + img_path[0]
  img_path = os.path.join(*img_path)
  # Now 'file_name' contains the name of the file
  return img_path

def split_data(categories_dirs: list, size, val = False):
  """
  This function returns two lists one for tuning samples annotations pathes
  and another for images pathes
  """
  annot_pathes = [
                        # getting the pathes of .xml file
                        os.path.join(categorie_dir,'Label',file_name)
                        for categorie_dir in categories_dirs
                        for file_name in os.listdir(categorie_dir + '/Label')[:math.ceil(len(os.listdir(categorie_dir + '/Label'))*size)]
                        # ensuring  the file extension is xml
                        if file_name.endswith('.txt')
                        ]

  images_pathes = [
                            get_images_path(annot_path)
                            for annot_path in annot_pathes
                        ]
  return annot_pathes, images_pathes