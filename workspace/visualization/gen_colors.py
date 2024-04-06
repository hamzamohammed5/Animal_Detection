import random
import os
import numpy as np

def generate_colors(classes_map: dict, colors_dir="",
                    R_min = 120,
                    R_max = 200,
                    G_min = 150,
                    G_max = 200,
                    B_min = 200,
                    B_max = 255
                    ):
  """generate a unique color for each class"""
  if os.path.exists(colors_dir+'/colors.npy'):
    colors = np.load(colors_dir+'/colors.npy',allow_pickle=True).item()
  else:
    colors = {}
    for i in  list(classes_map.keys()):
      color = (random.randint(R_min,R_max), random.randint(G_min,G_max), random.randint(B_min,B_max))
      # ensure the generated color is unique
      while color in list(colors.values()):
        color = (random.randint(120,200), random.randint(150,200), random.randint(200,255))
      # add in value color to key class in the colors dict dict
      colors[i] = color
    np.save(colors_dir+'/colors.npy', colors)
  return colors