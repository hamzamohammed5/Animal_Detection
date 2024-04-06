"""
Functions to fit the bbox and class text on object
"""

def handle_bbox_thickness(bbox):
    """
    these prameter has been assigned by tring and learn
    """
    x1, y1, x2, y2 = bbox
    p = 2* abs(x1 - x2) + 2* abs(y1 - y2 ) 

    if p<=500:
        box_thickness = 1
    elif p <= 1000:
        box_thickness = 2
    elif p <= 1500:
        box_thickness = 3
    elif p <= 2000:
        box_thickness = 4
    elif p <= 2500:
        box_thickness = 6
    elif p <= 3000:
        box_thickness = 8
    elif p <= 3500:
        box_thickness = 10
    elif p <= 4000:
        box_thickness = 12
    elif p <= 4500:
        box_thickness = 16
    else :
        box_thickness = 20
    return box_thickness

def fit_the_box(box_thickness, scale_value):
    """
    these prameter has been assigned by tring and learn
    """
    # handling left shifting
    # moveing the text up depends on the thickness of the bounding box
    c_left_shifting = box_thickness*scale_value/2

    # handling the background box hight to include all the text
    c_top_shifting = box_thickness*6
    c_top_shifting += c_top_shifting*scale_value/4
    

    # handling font sclae
    org_font_scale = box_thickness/6
    # scaling the font
    if scale_value >  1:
        font_scale   = org_font_scale + scale_value/10
    else:
        font_scale =  org_font_scale

    # handling text thickness of text
    if org_font_scale    < 0.5:
        text_thickness = 1
    elif org_font_scale  < 1:
        text_thickness = 2
    elif org_font_scale  < 1.5:
        text_thickness = 4
    elif org_font_scale  < 2:
        text_thickness = 6
    elif org_font_scale  < 2.5:
        text_thickness = 8
    else:
        text_thickness = 10
        
    # increase the thickness if the font scaled
    if scale_value    > 1:
        text_thickness  += scale_value/4

    # expand  the background box to the right to include the text
    c_expand_right = font_scale * 19

    # expand  the background box to the top to include the text
    c_expand_up = font_scale *40

    # if scale_value > 1:
    #   c_expand_up    += scale_value*c_expand_up/5

    return c_left_shifting, c_top_shifting, c_expand_right, c_expand_up, text_thickness, font_scale

def get_sample_data(sample,true_value, keys_list = ['image','classes','bboxes','scores'] ):
  """
  Extract classes and bboxes and score from our dataset
  """
  image   = sample[keys_list[0]]
  classes = sample[keys_list[2]]
  boxes  = sample[keys_list[1]]

  if true_value == False:
    # get the scores
    scores = sample[keys_list[3]]
    return image, classes, scores, boxes

  else:
    return image, classes, boxes
