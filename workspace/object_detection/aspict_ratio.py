import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from workspace.preprocessing import scale_with_padding, transform_xyxy_xywh


def average_iou(bboxes, anchors):
    """Calculates the Intersection over Union (IoU) between bounding boxes and
    anchors.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    anchors : Array of aspect ratios [n, 2] format.

    Returns:
    avg_iou_perc : A Float value, average of IOU scores from each aspect ratio
    """
    
    intersection_width = np.minimum(anchors[:, [0]], bboxes[:, 0]).T
    intersection_height = np.minimum(anchors[:, [1]], bboxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(bboxes, axis=1, keepdims=True)
    anchors_area = np.prod(anchors, axis=1, keepdims=True).T
    union_area = boxes_area + anchors_area - intersection_area
    avg_iou_perc = np.mean(np.max(intersection_area / union_area, axis=1)) * 100

    return avg_iou_perc


def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
  """Calculate the centroid of bounding boxes clusters using Kmeans algorithm.

  Args:
  bboxes : Array of bounding boxes in [width, height] format.
  kmeans_max_iter : Maximum number of iterations to find centroids.
  num_aspect_ratios : Number of centroids to optimize kmeans.

  Returns:
  aspect_ratios : Centroids of cluster (optmised for dataset).
  avg_iou_prec : Average score of bboxes intersecting with new aspect ratios.
  """

  assert len(bboxes), "You must provide bounding boxes"

  normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))
  
  # Using kmeans to find centroids of the width/height clusters
  kmeans = KMeans(
      init='random', n_clusters=num_aspect_ratios, random_state=0, max_iter=kmeans_max_iter)
  kmeans.fit(X=normalized_bboxes)
  ar = kmeans.cluster_centers_

  assert len(ar), "Unable to find k-means centroid, try increasing kmeans_max_iter."

  avg_iou_perc = average_iou(normalized_bboxes, ar)

  if not np.isfinite(avg_iou_perc):
    sys.exit("Failed to get aspect ratios due to numerical errors in k-means")

  aspect_ratios = [w/h for w,h in ar]

  return aspect_ratios, avg_iou_perc

def tune_aspect_ratios(data_sets:list, size: int, key_list = ['image', 'boxes', 'classes_ids'], kmeans_max_iter = 500, num_aspect_ratios = [1,2,3,4,5,6,7,8,9,10]):
  """
  Args:
  size: the Fixed input size for the mode
  
  Return:
  aspect ratios dictionay with number of aspect ratios, aspect ratios and iou for the aspect ratios with all boxes
  
  Normalized average area (devided by size ** 2)
  """
  scaled_boxes = []
  # collect bboces from datasets
  for data_set in data_sets:
    sizes = [size for i in range(len(data_set))]
    scaled_boxes += list(map(scale_with_padding, data_set[key_list[0]], data_set[key_list[1]], sizes))
  scaled_boxes = [_scaled_boxes[0] for _scaled_boxes in scaled_boxes]
  # convert to xywh
  scaled_xywh = list(map(transform_xyxy_xywh,scaled_boxes))

  # get w and h
  bboxes_W_H = [bbs[2:] for sample in  scaled_xywh for bbs in sample]
  bboxes_W_H = np.array(bboxes_W_H)

  # estimating average areas
  bboxes_A = bboxes_W_H[:,0]* bboxes_W_H[:,1]
  bboxes_A = bboxes_A.reshape(-1,1)
  kmeans = KMeans(
      init='random', n_clusters=1, random_state=0, max_iter=500)
  kmeans.fit(X=bboxes_A)
  Area = kmeans.cluster_centers_ / size**2
  
  aspect_ratios_dict = {}
  for num_ar in num_aspect_ratios: 
   aspect_ratios, avg_iou_perc = kmeans_aspect_ratios(bboxes_W_H, kmeans_max_iter, num_ar )
  
   aspect_ratios_dict[f'{num_ar}'] = {'aspect_ratios':aspect_ratios, 'iou': avg_iou_perc}

  x_axis = [num_ar for num_ar in num_aspect_ratios]
  y_axis = [aspect_ratios_dict[f'{num_ar}']['iou'] for num_ar in num_aspect_ratios]
  plt.plot(x_axis, y_axis)
  plt.scatter(x_axis, y_axis)
  plt.title("Mean IoU Score")
  plt.xlabel("Number of Aspect Ratios")
  plt.ylabel("IoU score")

  return aspect_ratios_dict, Area
