import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from workspace import preprocessing

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_feature_list(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(image, shape, xmins, ymins, xmaxs, ymaxs, classes_ids):
  image_format = b'jpg'
  feature = {
                'image/height':             int64_feature([shape[0]]),
                'image/width':              int64_feature([shape[1]]),
                'image/encoded':            image_feature(image),
                'image/format':             bytes_feature(image_format),
                'image/object/bbox/xmin':   float_feature(xmins), # the whole boxes xmin in image
                'image/object/bbox/xmax':   float_feature(xmaxs),
                'image/object/bbox/ymin':   float_feature(ymins),
                'image/object/bbox/ymax':   float_feature(ymaxs),
                'image/object/class/label': int64_feature(classes_ids),                                           # classes
                }


  return tf.train.Example(features=tf.train.Features(feature=feature))
  

def TFRedcord_wrtiter(df, tfrecords_dir, num_tfrecords, num_samples, split, start_rec = 0):
    if len(df) % (num_tfrecords - start_rec):
        num_tfrecords += 1  # add one record if there are any remaining samples
    # start writing tfrecords files
    for tfrec_num in range(start_rec, num_tfrecords):
      # iterate over ranges in  df
        samples = df.iloc[tfrec_num * num_samples: (tfrec_num + 1) * num_samples, :]
        samples['image'] = samples['image'].map(preprocessing.load_image)
        
        with tf.io.TFRecordWriter(
                   tfrecords_dir+ f"/{split}_%.{len(str(num_tfrecords))}i-%i.tfrecord" % (tfrec_num, len(samples)) ) as writer:
            # write the samples in the same tfrecord file
            for i in range(len(samples)):
                image         = samples['image'].iloc[i]
                bboxes        = samples['boxes'].iloc[i]
                classes_ids   = samples['classes_ids'].iloc[i]
                size          = image.shape
                xmins         = [float(bboxes[i][0]/size[1]) for i in range(len(bboxes))] # deviding x by width to normalize it
                ymins         = [float(bboxes[i][1]/size[0]) for i in range(len(bboxes))] # deviding y by height to normalize it
                xmaxs         = [float(bboxes[i][2]/size[1]) for i in range(len(bboxes))]
                ymaxs         = [float(bboxes[i][3]/size[0]) for i in range(len(bboxes))]
                example       = create_example(image, size, xmins, ymins, xmaxs, ymaxs, classes_ids)
                writer.write(example.SerializeToString())

def parse_tfrecord_fn(example):
    """
    return image, classes and bboxes in tensors
    """
    feature_description = {
                          'image/encoded':            tf.io.FixedLenFeature([], tf.string),
                          'image/object/bbox/xmin':   tf.io.VarLenFeature(tf.float32), # the whole boxes xmin in image
                          'image/object/bbox/xmax':   tf.io.VarLenFeature(tf.float32),
                          'image/object/bbox/ymin':   tf.io.VarLenFeature(tf.float32),
                          'image/object/bbox/ymax':   tf.io.VarLenFeature(tf.float32),
                          'image/object/class/label': tf.io.VarLenFeature(tf.int64)                                          # classes
                }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def process_dataset(example):
    # decode the image
    image_tn = tf.io.decode_jpeg(example["image/encoded"], channels=3)
    xmin  = tf.reshape(tf.sparse.to_dense(example['image/object/bbox/xmin']),(-1,1))
    ymin  = tf.reshape(tf.sparse.to_dense(example['image/object/bbox/ymin']),(-1,1))
    xmax  = tf.reshape(tf.sparse.to_dense(example['image/object/bbox/xmax']),(-1,1))
    ymax  = tf.reshape(tf.sparse.to_dense(example['image/object/bbox/ymax']),(-1,1))
    bboxes_tn = tf.concat([xmin,ymin, xmax,ymax], axis=1)
    classes_tn = tf.sparse.to_dense(example['image/object/class/label'])
    return image_tn, bboxes_tn, classes_tn
