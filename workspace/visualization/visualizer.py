"""
```
# Example of using
# if each sample in your dataset is in this structure
sample = {
          'image'   :  image,
          'classes' : list_of_classes, 
          'bboxes'  : list_of_bboxes_in_xyxy_format,
          'scores'  : list_of_scores,
          }
} 
# your keys_list sould be like this
keys_list = ['image','classes','bboxes','scores']


```
-Note: the keys_list items should be in 
this order mentioned abouve even if 
the order changes the sample dictionary.

-You can handle any dataset by editting 
in the handle_datasets function in handle_datasets file to return 
a list of samples, and each sample should be in the required strure, 
and then an argument using dataset_info parameter 
to it to handle many conditions.

-You can draw each sample separatly and 
save it using draw_samples function in draw file.

"""
import numpy as np
from matplotlib import pyplot as plt
import random
import draw  

# to visualize images in notebook run these 2 lines
# %matplotlib notebook
# %matplotlib inline


def visualize_samples(dataset, no_samples, nrows, ncols, classes_map,
                      True_value      = True, 
                      dataset_info    = '',
                      scale_thickness = 1, 
                      keys_list       = ['image','classes','bboxes','scores'],
                      figsize         = (25, 25),
                      title           = None,
                      title_size      = 50,
                      colors_dir      = '',
                      save            = False,
                      saving_path     = '',
                      separation      = (0,0)
                      ):
  # get a list of images with drawn bboxes and classes
  images = draw.draw_samples(dataset,
                             no_samples      = no_samples, 
                             classes_map     = classes_map,
                             True_value      = True_value, 
                             scale_thickness = scale_thickness, 
                             keys_list       = keys_list, 
                             colors_dir      = colors_dir,
                             dataset_info    = dataset_info
                             )
  # initialize the fig
  fig = plt.figure(figsize = figsize)
  fig.suptitle(title, size = title_size)
  plt.subplots_adjust(wspace=separation[0], hspace=separation[1])
  plt.tight_layout()
  # plotting the images in the fig
  for i, image in enumerate(images):
    fig.add_subplot(nrows, ncols, i+1)
    plt.axis('off')
    
    # to show image if its values from 0 to 1
    # image = np.clip(img_list[i], 0, 1)
    
    plt.imshow(image.astype(int))
  # disable automatic image showing
  plt.close(None)

  # saving
  if save == True:
    plt.savefig(saving_path, bbox_inches='tight')

  return fig

def vis_comparison(truths, predictions,
               figsize = (12,9),
               separation = (0,0),
               nrows = 2,
               ncols = 2,
               samples_per_image = 2,
               title_size = 50,
               title      = None,
               save = False,
               saving_path = '',
               keys_list = ['image', 'boxes', 'classes', 'scores']
               ):
  figs = []
  if save == True:
      if not os.path.exists(saving_path):
           os.makedirs(saving_path)

  for i in range(math.ceil(len(truths)/samples_per_image)):
    fig = plt.figure(figsize = figsize)
    fig.suptitle(title, size = title_size)
    plt.subplots_adjust(wspace=separation[0], hspace=separation[1])
    plt.tight_layout()
    idx = 1

    # plotting the images in the fig
    for truth, prediction in zip(truths[i*samples_per_image:(i+1)*samples_per_image], predictions[i*samples_per_image:(i+1)*samples_per_image]):
      fig.add_subplot(nrows, ncols, idx)
      plt.axis('off')
      plt.imshow(truth.astype(int))
      idx +=1

      fig.add_subplot(nrows, ncols, idx)
      plt.axis('off')
      plt.imshow(prediction.astype(int))
      idx +=1
    plt.show()
    figs.append(fig)

  if save == True:
    for i,fig in enumerate(figs):
      fig.savefig(f'{saving_path}/{i+1}.jpg', bbox_inches='tight')