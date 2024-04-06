import processing
import cv2
import gen_colors
import handle_datasets
import numpy as np

def draw_predicted_class(img, bbox, box_thickness, 
                          label = list, 
                          score= float,
                          Text_color = (0,0,0), 
                          background_color = (255,0,0), 
                          scale_thickness = 1
                          ):
    """
    plot the predicted class and its score  on the object
    the background box coordinates
    """
    text = f'{str(score*100)[:5]}% {label}'
    c_left_shifting, c_top_shifting, c_expand_right, c_expand_up, text_thickness, font_scale = processing.fit_the_box(box_thickness, scale_thickness)
    
    x1, y1, _,_ = bbox
    # subtract in y1 to move the text up the bbox and in x1 to move the text to left, 
    # and in y2 to expand the background box to top
    if y1 - c_expand_up < 0:
        y2 = y1 + c_expand_up + scale_thickness
        x1 = x1 
        x2 = x1 + len(text)*c_expand_right+c_expand_right

        y1_text = y1+c_top_shifting/2

    else:
      """
    -------(x2,y2)
    |             |
    |             |
    |             |
    (x1,y1)--------
      """
      x1 = x1 - c_left_shifting
      y1 = y1 - c_left_shifting 
      x2 = x1 + len(text)*c_expand_right+c_expand_right
      y2 = y1 - c_expand_up 

      y1_text = y1-c_top_shifting/4
    
    # We will put the thick by negative to fill it
    # y1++ to include all the text
    img = cv2.rectangle(img, (int(x1), int(y1+c_top_shifting/6.75)), (int(x2+5),int(y2)), color = background_color, thickness=-1)

    img = cv2.putText(img, text, 
                        org       = (int(x1+5),int(y1_text)), 
                        color     = Text_color, 
                        thickness = int(text_thickness),
                        fontFace  = cv2.FONT_HERSHEY_COMPLEX,
                        fontScale = font_scale 
                        )
    return img
    
def draw_true_class(img, bbox, box_thickness, 
                    label = list, 
                    Text_color = (0,0,0), 
                    background_color = (255,0,0), 
                    alpha = 0.5, 
                    scale_thickness = 1
                    ):
    """
    plot the True class on the object
    the background box coordinates
    """
    text = label
    c_left_shifting, c_top_shifting, c_expand_right, c_expand_up, text_thickness, font_scale = processing.fit_the_box(box_thickness, scale_thickness)

    x1, y1, _,_ = bbox
    # subtract in y1 to move the text up the bbox and in x1 to move the text to left, 
    # and in y2 to expand the background box to top
    if y1 - c_expand_up < 0:
        y2 = y1 + c_expand_up + scale_thickness
        x1 = x1 
        x2 = x1 + len(text)*c_expand_right+c_expand_right

        y1_text = y1+c_top_shifting/2
    else:
      """
    -------(x2,y2)
    |             |
    |             |
    |             |
    (x1,y1)--------
      """
      x1 = x1 - c_left_shifting
      y1 = y1 - c_left_shifting 
      x2 = x1 + len(text)*c_expand_right+c_expand_right
      y2 = y1 - c_expand_up

      y1_text = y1-c_top_shifting/4
    
    
    # We will put the thick by negative to fill it
    # y1++ to include all the text
    img = cv2.rectangle(img, (int(x1), int(y1+c_top_shifting/6.75)), (int(x2+5),int(y2)), color = background_color, thickness=-1)

    img = cv2.putText(img, text, 
                        org       = (int(x1+5),int(y1_text)), 
                        color     = Text_color, 
                        thickness = int(text_thickness),
                        fontFace  = cv2.FONT_HERSHEY_COMPLEX,
                        fontScale = font_scale 
                        )
    return img


def draw_bbox(img: float, bbox: list, box_color, scale_thickness = 1):
  """
  the  bbox coordinates
  (x1,y1)-------
  |             |
  |             |
  |             |
  ----------(x2,y2)
  """
  x1, y1, x2, y2 = bbox
  # handling the bboxe tickness
  box_thickness  = processing.handle_bbox_thickness(bbox)
  bbox_thickness = box_thickness*scale_thickness

  start_point = (int(x1), int(y1))
  end_point = (int(x2), int(y2))
  
  img = cv2.rectangle(img, start_point, end_point, color=box_color, thickness=bbox_thickness)
  return img, box_thickness

def draw_on_image(image, boxes, classes_names, colors,
                  scores = None, 
                  True_value = False,
                  alpha = 0.5,
                  scale_thickness = 1
                  ):
  """
  To draw the bounding boxes and classes for the objects in one image
  """
  img = image.copy()
  for i in range(len(classes_names)):
      image, bbox_thick = draw_bbox(image, boxes[i], 
                                    box_color       = colors[classes_names[i]] , 
                                    scale_thickness = scale_thickness)
      if True_value== True:
        image = draw_true_class(image, 
                                boxes[i], bbox_thick, classes_names[i], 
                                background_color = colors[classes_names[i]],
                                scale_thickness  = scale_thickness)
      else:
        image = draw_predicted_class(image, 
                                    boxes[i], bbox_thick, classes_names[i], scores[i],
                                    background_color = colors[classes_names[i]],
                                    scale_thickness  = scale_thickness
                                    )
  image = cv2.addWeighted(image, alpha, img, 1 - alpha, 0)
  return image


def draw_sample(sample, colors,True_value, classes_map, 
                keys_list       = ['image','bboxes','classes','scores'], 
                alpha = 0.5,
                Scale_thickness = 1
                ):
  """
  To draw the bounding boxes and classes for the objects in one image (sample)
  """
  # parse image, classes, bboxes and scores from the sample dict
  if True_value == True:
    img, classes, boxes = processing.get_sample_data(sample, True_value, keys_list= keys_list)
    scores = None
  else:
    img, classes , scores, boxes = processing.get_sample_data(sample, True_value, keys_list= keys_list)
  image = np.copy(img)
  # convert classes from integers to string
  classes_names= [name for i in classes for name ,ind in classes_map.items()  if ind == i ]
    
  # start drawing the boxes and its classes
  image = draw_on_image(image, boxes, classes_names, 
                        scores          = scores, 
                        True_value      = True_value,
                        colors          = colors,
                        alpha = alpha,
                        scale_thickness = Scale_thickness
                        )
      
  return image

def draw_samples(dataset, no_samples, True_value:bool, classes_map: dict,
                 save            = False, 
                 savind_dir      = '',
                 colors_dir      = '', 
                 alpha           = 0.5,
                 scale_thickness = 1, 
                 keys_list       = ['image','bboxes','classes','scores'], 
                 dataset_info    = ''
                 ):
  """
  To draw the bboxes and classes for the objects in no_samples images
  """
  # generating a unique color  for each class
  colors = gen_colors.generate_colors(classes_map,colors_dir)
  # getting a list of samples in a dict in the required structure 
  samples = handle_datasets.handle_datasets(dataset, no_samples, True_value, keys_list , dataset_info = dataset_info)

  # images with a drawn bboxes and classes
  images = []

  # iterate over each sample in the list and draw its bbox and class
  for sample in samples:
    image = draw_sample(sample, colors, 
                        True_value      = True_value, 
                        classes_map     = classes_map, 
                        alpha           = alpha,
                        Scale_thickness = scale_thickness, 
                        keys_list       = keys_list
                        )
    images.append(image)

  # saving the image 
  if save == True:
    for i in range(len(images)):
      cv2.imwrite(f'{savind_dir}/image{i}.jpg', cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
  return images
  