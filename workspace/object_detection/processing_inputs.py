import tensorflow as tf
from util import swap_xy, random_flip_horizontal, resize_and_pad_image, convert_to_xywh
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, ReLU

min_side = 124.0
max_side = 256.0
jitter = [124, 200]
stride = 32

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, max_side=max_side, min_side=min_side ,jitter=None, stride=stride)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0)/255.0, ratio



class process_ds():
  """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """

  def __init__(self,min_side, max_side, jitter, stride):
    self.min_side = min_side
    self.max_side = max_side
    self.jitter = jitter
    self.stride = stride

  def __call__(self, image, bbox, class_id) :
    image = tf.cast(image,tf.float32)
    bbox  = tf.cast(bbox, tf.float32)
    class_id = tf.cast(class_id, tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image, self.min_side, self.max_side, self.jitter, self.stride)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image/255.0, bbox, class_id

class head():
  def __init__(self, blocks):
    self.blocks = blocks

  def build_head(self, output_filters, name, bias_init):
      """Builds the class/box predictions head.

      Arguments:
        output_filters: Number of convolution filters in the final layer.
        bias_init: Bias Initializer for the final convolution layer.

      Returns:
        A keras sequential model representing either the classification
          or the box regression head depending on `output_filters`.
      """
      head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, 256])])
      kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
      for _ in range(self.blocks):
          head.add(
              tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init, name = f'conv_{_}/{name}')
          )

          head.add(tf.keras.layers.ReLU())
      head.add(
          tf.keras.layers.Conv2D(
              output_filters,
              3,
              1,
              padding="same",
              kernel_initializer=kernel_init,
              bias_initializer=bias_init,
              name = f'{name}_head'
          )
      )
      return head



