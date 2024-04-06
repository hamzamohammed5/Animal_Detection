import tensorflow as tf

def  handle_datasets(dataset,no_samples,true_value, 
                     keys_list = ['image','bboxes','classes','scores'], 
                     dataset_info=''
                     ):
  """
  convert no_samples samples from the dataset to the required structure to fit into the visualizer

  return list of smples
  """
  if dataset_info == "tf":
    samples_lst = tf_dataset(dataset, no_samples, true_value, keys_list)
  elif dataset_info == "df":
    samples_lst = pd_DataFrame(dataset, no_samples, true_value, keys_list)
  elif dataset_info == "test":
    samples_lst = testing_list(dataset, no_samples, true_value, keys_list)
  return samples_lst

def pd_DataFrame(df,no_samples,true_value,
                 keys_list = ['image','bboxes','classes','scores']
                 ):
    samples = df.iloc[0:no_samples, :]
    images = samples[keys_list[0]].values.tolist()
    bboxes = samples[keys_list[1]].values.tolist()
    classes = samples[keys_list[2]].values.tolist()
    samples_lst = []
    for i in range(0,no_samples):
      sample_dict = {keys_list[0]: images[i],
                     keys_list[2]: classes[i],
                     # convert the values of bboxes to int
                     keys_list[1]: [list(map(int,i)) for i in bboxes[i]]
                     }
      samples_lst.append(sample_dict)
    return samples_lst

def tf_dataset(dataset,no_samples,true_value,
               keys_list = ['image','bboxes','classes','scores']
               ):
  """
  convert no_samples samples from the dataset to the required structure to fit into the visualizer

  return list of smples
  """
  samples     = dataset.take(no_samples)
  samples     = samples.as_numpy_iterator()
  samples_lst = []

  # iterate over the required sampls in the data set
  # and extract image array, classes, bboxes and scores if exist
  for i in range(0,no_samples):
    sample = samples.next()
    sample_dict = {keys_list[0]: tf.keras.backend.get_value(sample[0]),        # image                                                        # objects
                   keys_list[2]: sample[1][keys_list[1]].astype(int).tolist(), # classes
                   keys_list[1]: sample[1][keys_list[2]].astype(int).tolist(), # bboxes             
                   }
    # adding score
    if true_value == False:
      sample_dict[keys_list[3]] = sample[1][keys_list[3]].tolist()

    samples_lst.append(sample_dict)

  return samples_lst

def testing_list(dataset,no_samples,true_value,
               keys_list = ['image','bboxes','classes','scores']
):
  images_lst, boxes_lst, classes_lst, scores_lst = dataset
  samples_lst = []
  for image, boxes, classes, scores in zip(images_lst, boxes_lst, classes_lst, scores_lst):
      sample_dict = {keys_list[0]: image,
                     keys_list[2]: [int(c) for c in classes],
                     # convert the values of bboxes to int
                     keys_list[1]: boxes,
                     keys_list[3]: [round(score,3) for score in scores]
                     }
      samples_lst.append(sample_dict)
  return samples_lst
      
