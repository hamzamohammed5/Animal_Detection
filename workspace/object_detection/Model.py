import math

import tensorflow as tf
import numpy as np
import pandas as pd

from processing_inputs import prepare_image
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, ReLU
from workspace.visualization import visualizer

from tensorflow import keras



class MyModel(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, FeaturePyramid, build_head, input_processor, num_classes, num_anchors, process_input = True,**kwargs):
       
        super().__init__(name="MyModel", **kwargs)
        self.process_input = process_input
        self.input_processor = input_processor
        self.fpn = FeaturePyramid
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(num_anchors * num_classes, 'classification', prior_probability)
        self.box_head = build_head(num_anchors * 4, 'box', "zeros")

    def call(self, image, training=False):
        features = self.fpn(image)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)
    
    # def get_input_processor(self):
    #     return self.process_input, self.input_processor
        

def build_mymodel(Model, process_input = True):
    fpn, head, input_processor, num_classes, num_anchors = Model()
    model = MyModel(fpn, head, input_processor, num_classes, num_anchors, process_input)
    return model

def build_my_inference_model(mymodel, input_shape, Decoder):
    
    image = tf.keras.Input(shape=input_shape, name="image")
    predictions = mymodel(image, training=False)
    detections = Decoder(image, predictions)
    
    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    return inference_model

def plot_model_metic(models_names, path, string='loss', skip_epochs=None):
  """Helper function to plot model metric for train an val in the same graph"""
  for model_name in models_names:
    logs = path + f'/{model_name}/logs.csv'
    history = pd.read_csv(logs)
    plt.plot(history[string][skip_epochs:])
    plt.plot(history['val_'+string][skip_epochs:])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.title(model_name)
    plt.show()

def test_image(img_path, iference_model, classes_map, colors_dict,
               scale_thickness = 1,
               print_detection = False,
               keys_list = ['image', 'boxes', 'classes', 'scores']
               ):
  # preocessing
  image = preprocessing.load_image(img_path)
  input_image, ratio = prepare_image(image)
  # inferencing
  detections = iference_model.predict(input_image)
  if print_detection:
    print(detections)
  num_detections = detections.valid_detections[0]
  bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
  classes = detections.nmsed_classes[0][:num_detections]
  scores = detections.nmsed_scores[0][:num_detections].tolist()

  # preparing for drawing
  sample_dict = {keys_list[0]: image,
                     keys_list[2]: [int(c) for c in classes],
                     # convert the values of bboxes to int
                     keys_list[1]: bboxes,
                     keys_list[3]: [round(score,3) for score in scores]
                     }

  reverse_dict = {v: k for k, v in classes_map.items()}
  print(f"prediction: ",[[round(score*100,1), reverse_dict[cls]] for score, cls in zip(scores, classes.astype(int).tolist())])
  print(classes)
  new_image = draw.draw_sample(sample_dict, colors_dict,
                   True_value      = False,
                   classes_map     = classes_map,
                   Scale_thickness = scale_thickness,
                   keys_list       = keys_list
                        )
  return new_image

def fast_box_testing(test_ds, iference_model, classes_map,
                     print_detection = False,
                     print_prediction = False,
                     print_truth = False,
                     num_samples = 10,
                     skip_samples = 0
                     ):
  reverse_dict = {v: k for k, v in classes_map.items()}
  num_detections_lst = []
  for i, sample in enumerate(test_ds.skip(skip_samples).take(num_samples)):
      image = sample[0]
      print(image.shape)
      input_image, ratio = prepare_image(image)
      detections = iference_model.predict(input_image)
      if print_detection:
        print(detections)
      num_detections = detections.valid_detections[0]
      num_detections_lst.append(num_detections)
      bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
      classes = detections.nmsed_classes[0][:num_detections]
      scores = detections.nmsed_scores[0][:num_detections].tolist()

      bbb = bboxes.numpy().astype(int).tolist()
      p = image.numpy()
      for bb in bbb:
        p,_ = visualizer.draw.draw_bbox(p, bb, (255,0,0),1)
      plt.imshow(p.astype(int))
      plt.show()
      if print_prediction:
        print(f"prediction {i+1}: ",[[round(score*100,1), reverse_dict[cls]] for score, cls in zip(scores, classes.astype(int).tolist())])
      if print_truth:
        print(f"Truth: ", [reverse_dict[cls] for cls in sample[2].numpy().tolist()])

def calculate_speed(pathes:list, models_lst: list, tf_record, num_classes):
  test_ds = get_testing_dataset(tf_record)

  for model, path in zip(models_lst, pathes):
    print(f'Calculating speed for {path.split(os.path.sep)[-1]}')
    model_weights = path + '/weights.h5'

    mymodel = model()
    MyModel = Model.build_mymodel(mymodel)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate= 0.0025, momentum=0.9)
    loss_fn = loss.MyModelLoss(num_classes = num_classes)
    MyModel.compile(loss=loss_fn, optimizer=optimizer)
    MyModel(tf.zeros((1,224,224,3)))
    MyModel.load_weights(model_weights)
    encoder = Coder.LabelEncoder(mymodel.get_anchor_gen(), mymodel.get_box_variance(), mymodel.get_input_processor(), process_input=False)
    decoder = Coder.DecodePredictions(mymodel.get_anchor_gen(), mymodel.get_box_variance(), num_classes,nms_iou_threshold=0.5, confidence_threshold=0.2)
    inference_model = Model.build_my_inference_model(MyModel, (None,None,3), decoder)

    speed = []
    itr_test_ds = test_ds.as_numpy_iterator()
    while len(speed)<20 :
      input_image, ratio = prepare_image(itr_test_ds.next()[0])
      start = time.time()
      detections = inference_model.predict(input_image)
      end = time.time()
      inf_speed = end - start
      if inf_speed < 1:
        speed.append(inf_speed)

    hist_path = path + '/hist.npy'
    if os.path.exists(hist_path):
      hist = np.load(hist_path,allow_pickle=True).item()
    else:
      hist = {}

    hist['speed'] = speed
    print(f'{path.split(os.path.sep)[-1]} speed: ',speed)
    np.save(hist_path, hist)


def do_train(model_dir,
             MyModel,
             loss,
             optimizer,
             train_ds,
             val_ds,
             weights = None,
             training = True,
             continue_training = True,
             epochs = 100,
             train_steps = None,
             val_steps = None,
             early_stop = 5,
             callbacks = None):

  best_loss = None
  skip_epochs = None
  epoch_num = None
  patient = 0

  if weights:
    MyModel(tf.zeros((1,224,224,3)))
    MyModel.load_weights(weights)
    if os.path.exists(f'{model_dir}/logs.csv'):
      logs = pd.read_csv(f'{model_dir}/logs.csv')
      skip_epochs = logs['epoch'].iloc[-1]
      best_loss   = float(logs['loss'].min())
      if continue_training:
        patient = skip_epochs - int(logs[logs['loss'] == best_loss]['epoch'])

  for i in range(epochs):
    if not training:
      break
    if skip_epochs and skip_epochs >= i:
      continue

    hist = MyModel.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=i+1,
        initial_epoch = i,
        callbacks = callbacks
        )
    trial = 0
    while np.isnan(hist.history['loss'][0]) and trial < 2:
      print('Handling Overshooting: reducing learning rate by /2')
      MyModel.load_weights(f"{model_dir}/weights.h5")
      lr = optimizer.learning_rate.numpy()
      optimizer.learning_rate.assign(lr/5.0)
      MyModel.compile(loss=loss, optimizer=optimizer)
      pd.read_csv(f"{model_dir}/logs.csv").iloc[:-1,:].to_csv(f"{model_dir}/logs.csv")

      hist = MyModel.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=i+1,
        initial_epoch = i,
        callbacks = callbacks
        )
      trial += 1

    if trial == 2:
      print('Overshooting')
      break

    if not best_loss:
      best_loss = float(hist.history['loss'][0])
      MyModel.save_weights(f"{model_dir}/weights.h5")
      print(f"weights saved in {model_dir}")
      epoch_num = i+1

    if float(hist.history['loss'][0]) < best_loss:
      best_loss = float(hist.history['loss'][0])
      MyModel.save_weights(f"{model_dir}/weights.h5")
      print(f"weights saved in {model_dir}")
      patient = 0
      epoch_num = i+1

    else:
      patient += 1
      if patient == early_stop:
        print(f"Best weights: epoch {epoch_num}, loss = {best_loss}")
        break


def train_model(model, model_weights, continue_training, train_rec, val_rec, test_rec, num_classes, classes_map, colors_dir,
                training = True,
                model_dir = '/content/drive/MyDrive/My_Model',
                train_steps = None,
                val_steps = None ,
                BATCH_SIZE = 2,
                early_stop = 3,
                learning_rate = 0.0025,
                epochs = 100,
                nms_confidence_threshold = 0.5,
                nms_iou_threshold = 0.5
                ):

    tf.keras.backend.clear_session()
    csv_logger= CSVLogger(filename=f'{model_dir}/logs.csv', separator=",", append= True)
    callbacks = csv_logger

    mymodel = model()
    MyModel = Model.build_mymodel(mymodel)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate= learning_rate, momentum=0.9)
    loss_fn = loss.MyModelLoss(num_classes = num_classes)
    MyModel.compile(loss=loss_fn, optimizer=optimizer)
    encoder = Coder.LabelEncoder(mymodel.get_anchor_gen(), mymodel.get_box_variance(), mymodel.get_input_processor(), process_input=False)
    decoder = Coder.DecodePredictions(mymodel.get_anchor_gen(), mymodel.get_box_variance(), num_classes, nms_iou_threshold=nms_iou_threshold, confidence_threshold=nms_confidence_threshold)
    train_ds = get_dataset(train_rec, BATCH_SIZE, mymodel.get_ds_processor(), encoder)
    val_ds = get_dataset(val_rec, BATCH_SIZE, mymodel.get_ds_processor(), encoder)

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    print(f'Start training {model_dir.split(os.path.sep)[-1]}')
    if not train_steps:
      train_steps = train_ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()
    if not val_steps:
      val_steps   = val_ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()

    train_ds = train_ds.prefetch(buffer_size= tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size= tf.data.AUTOTUNE)

    # start training
    do_train(model_dir, MyModel, loss_fn, optimizer, train_ds, val_ds, model_weights, training, continue_training, epochs, train_steps, val_steps, early_stop, callbacks)

    MyModel.load_weights(f"{model_dir}/weights.h5")
    inference_model = Model.build_my_inference_model(MyModel, (None,None,3), decoder)
    test_ds = get_testing_dataset(test_rec)

    # test and save model prediction
    print("Start Testing")
    test_tuned(test_ds, inference_model, 20, classes_map, f"{model_dir}/prediction.jpg", colors_dir, scale_thickness= 5, figsize=(25,25), separation=(0,0))

    # testing model speed
    print(f"Testing {model_dir.split(os.path.sep)[-1]} speed")
    calculate_speed([model_dir] , [model], train_rec)

    hist_path = f'{model_dir}/hist.npy'
    hist = np.load(hist_path,allow_pickle=True).item()

    pred_boxes,  true_boxes = handle_ds(test_ds, inference_model)
    print(f"Calculating mAP for {model_dir.split(os.path.sep)[-1]}")
    PR_curves = model_dir + '/PR_curves'
    if not os.path.exists(PR_curves):
      os.makedirs(PR_curves)
    mAP = calculate_mAP(pred_boxes, true_boxes, num_classes, classes_map, PR_curves, np.arange(0.5, 0.95  + 0.05, 0.05, dtype=float))
    print('mAP: ',mAP)
    hist['mAP'] = mAP
    np.save(hist_path, hist)


def tune_models(models_lst, models_weights, training_lst, models_num, continue_training_list, train_rec, val_rec, num_classes, classes_map,
                train_steps = None,
                val_steps = None ,
                BATCH_SIZE = 2,
                early_stop = 5,
                learning_rate = 0.0025,
                epochs = 100,
                nms_confidence_threshold = 0.2):

  early_stop = early_stop
  main_dir = '/content/drive/MyDrive/tuning'

  if not os.path.exists(main_dir):
    os.makedirs(main_dir)

  for name, model, weights, training, continue_training in zip(models_num, models_lst, models_weights, training_lst, continue_training_list):
    model_dir = f"{main_dir}/model_{name}"
    train_model(
                model         = model,
                training      = training,
                model_weights = weights,
                model_dir     = model_dir,
                train_rec     = train_rec,
                val_rec       = val_rec,
                test_rec      = train_rec,
                num_classes   = num_classes,
                classes_map   = classes_map,
                BATCH_SIZE    = BATCH_SIZE,
                train_steps   = train_steps,
                val_steps     = val_steps,
                early_stop    = early_stop,
                learning_rate = learning_rate,
                continue_training = continue_training,
                nms_confidence_threshold = nms_confidence_threshold
                )


def vis_models_tuning(models_names, main_path = '/content/drive/MyDrive/tuning'):
  models_tuning = []
  for model_name in models_names:
    logs = pd.read_csv(main_path+ f'/{model_name}/logs.csv')
    loss = logs['loss'].min()
    val_loss = logs[logs['loss'] == loss]['val_loss'].tolist()[0]
    epoch = logs[logs['loss'] == loss]['epoch'].tolist()[0] +1
    mAP = np.load(main_path+ f'/{model_name}/hist.npy',allow_pickle=True).item()['mAP']*100
    speed = np.load(main_path+ f'/{model_name}/hist.npy',allow_pickle=True).item()['speed']
    speed = sum(speed)/len(speed)
    models_tuning.append([model_name, epoch, loss, val_loss, mAP, round(speed,3)])
  df = pd.DataFrame(models_tuning, columns= ['Name', 'Epoch','loss', 'val_loss', 'mAP', 'speed'])
  return df

def test_tuned(ds,
               inference_model,
               no_samples,
               classes_map,
               saving_path,
               colors_dir,
               scale_thickness = 1,
               figsize = (10,10),
               separation = (1,1),
               keys_list = ['image','bboxes','classes','scores'],
               print_prediction = True,
               print_truth = True,
               show_fig = False
               ):
    reverse_dict = {v: k for k, v in classes_map.items()}

    images_lst = []
    boxes_lst =  []
    classes_lst = []
    scores_lst  = []
    num_detections_lst = []

    for i,sample in enumerate(ds.take(no_samples)):
        image = sample[0]
        input_image, ratio = prepare_image(image)
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        num_detections_lst.append(num_detections)
        bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
        classes = detections.nmsed_classes[0][:num_detections]
        scores = detections.nmsed_scores[0][:num_detections].tolist()
        bboxes = bboxes.numpy().astype(int).tolist()
        images_lst.append(image.numpy())
        boxes_lst.append(bboxes)
        classes_lst.append(classes)
        scores_lst.append(scores)
        if print_prediction:
          print(f"prediction {i+1}: ",[[round(score*100,1), reverse_dict[cls]] for score, cls in zip(scores, classes.astype(int).tolist())])
        if print_truth:
          print(f"Truth: ", [reverse_dict[cls] for cls in sample[2].numpy().tolist()])


    data = [images_lst, boxes_lst, classes_lst, scores_lst]

    # visualize resized data
    fig = visualizer.visualize_samples(data,
                                no_samples      = no_samples,
                                ncols           = 4,
                                nrows           = math.ceil(no_samples/4),
                                True_value      = False,
                                classes_map     = classes_map,
                                keys_list       = keys_list,
                                scale_thickness = scale_thickness,
                                figsize         = (10,10),
                                separation      = separation,
                                dataset_info    = 'test',
                                colors_dir      = colors_dir
                                )
    if show_fig:
      fig.show()
    fig.savefig(saving_path, bbox_inches='tight')

def Compare_detected(ds, inference_model, classes_map, no_samples, colors_dir,
                     skip_samples       = 0,
                     alpha              = 0.5,
                     scale_thickness    = 1,
                     figsize            = (12,9),
                     separation         = (0,0),
                     nrows              = 2,
                     ncols              = 2,
                     samples_per_image  = 2,
                     title_size         = 25,
                     title              = 'Truth/Prediction',
                     save               = False,
                     saving_path        = '',
                     keys_list          = ['image','objects','classes','bboxes','scores']
                     ):
  colors_dict = gen_colors.generate_colors(classes_map, colors_dir)
  truth      = []
  prediction = []

  for i,sample in enumerate(ds.skip(skip_samples).take(no_samples)):
    image = sample[0]
    shape = image.shape
    input_image, ratio = prepare_image(image)
    # inferencing
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
    classes = detections.nmsed_classes[0][:num_detections]
    scores = detections.nmsed_scores[0][:num_detections].tolist()

    # preparing for drawing
    sample_prediction = {keys_list[0]: image,
                        keys_list[2]: [int(c) for c in classes],
                        # convert the values of bboxes to int
                        keys_list[1]: bboxes,
                        keys_list[3]: [round(score,3) for score in scores]
                        }

    x_min, y_min, x_max, y_max = np.split(sample[1], 4, axis=1)
    x_min = x_min * shape[1]
    x_max = x_max * shape[1]
    y_min = y_min * shape[0]
    y_max = y_max * shape[0]
    true_boxes = np.concatenate([x_min, y_min, x_max, y_max], axis=1).astype(int).tolist()

    sample_truth = {
                        keys_list[0]: image,
                        keys_list[2]: [int(c) for c in sample[2]],
                        # convert the values of bboxes to int
                        keys_list[1]: true_boxes
                    }

    # reverse_dict = {v: k for k, v in classes_map.items()}
    # print(f"prediction: ",[[round(score*100,1), reverse_dict[cls]] for score, cls in zip(scores, classes.astype(int).tolist())])

    sample_prediction = draw.draw_sample(sample_prediction, colors_dict,
                      True_value      = False,
                      classes_map     = classes_map,
                      alpha = alpha,
                      Scale_thickness = scale_thickness,
                      keys_list       = keys_list
                          )
    sample_truth = draw.draw_sample(sample_truth, colors_dict,
                      True_value      = True,
                      classes_map     = classes_map,
                      alpha = alpha,
                      Scale_thickness = scale_thickness,
                      keys_list       = keys_list
                          )
    truth.append(sample_truth)
    prediction.append(sample_prediction)
    
  vis_comparison(truth, prediction, figsize = figsize, separation = separation,
                 nrows = nrows,
                 ncols = ncols,
                 samples_per_image = samples_per_image,
                 title_size = title_size,
                 title      = title,
                 save = save,
                 saving_path = saving_path,
                 keys_list = keys_list
                 )
  
def test_on_videos(sources:list,
                   output_video,
                   colors_dir,
                   classes_map,
                   inference_model,
                   alpha = 0.5,
                   scale_thickness = 5):
  
  result = None
  colors_dict = gen_colors.generate_colors(classes_map, colors_dir)
  reverse_dict = {v: k for k, v in classes_map.items()}
  for source in sources:
    print(f'working on: {source}')
    video = cv2.VideoCapture(source)
    if (video.isOpened() == False):
        print("Error reading video file")
    if not result:
      size = (720, 1280)
      fps = video.get(cv2.CAP_PROP_FPS)
      result = cv2.VideoWriter('output_video',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, size)
    while video.isOpened():
      ret, image = video.read()
      if ret:
        # detect
        input_image, ratio = prepare_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # inferencing
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
        classes = detections.nmsed_classes[0][:num_detections]
        scores = detections.nmsed_scores[0][:num_detections].tolist()

        classes = [reverse_dict[cls] for cls in classes]
        scores = [round(score,3) for score in scores]

        # render detection
        frame_out = image.copy()
        for box, class_, score in zip(bboxes, classes, scores):
          frame_out, bbox_thick = draw.draw_bbox(frame_out, box,
                                        box_color       = colors_dict[class_] ,
                                        scale_thickness = scale_thickness)
          frame_out = draw.draw_predicted_class(frame_out,
                                        box, bbox_thick, class_, score,
                                        background_color = colors_dict[class_],
                                        scale_thickness  = scale_thickness
                                        )
        frame_out = cv2.addWeighted(frame_out, alpha, image, 1 - alpha, 0)
        # write the video
        if frame_out.size != size:
          frame_out = cv2.resize(frame_out, size)
        result.write(frame_out)
        if cv2.waitKey(1) & 0xFF == ord('x'):
          result.release()
          break
      else: break
    video.release()
  result.release()
  print('done!')
  

def lateral_connection(bottom_up_feature, top_down_feature, name):
    p = Conv2D(256, 1, 1, 'same', name = name)(bottom_up_feature) + UpSampling2D(2)(top_down_feature)
    return p

class head2():
  def __init__(self, blocks):
    self.blocks = blocks

  def build_head(self, output_filters, name, bias_init):
      """Builds the normalized class/box predictions head.

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

      head.add(BatchNormalization(axis = 3, name = f'{name}_head/bn'))

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
  
class head3():
  def __init__(self, blocks):
    self.blocks = blocks

def build_head(self, output_filters, name, bias_init, blocks = 4):
    """Builds the normalized class/box predictions head.

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
        head.add(BatchNormalization(axis = 3, name =  f'conv_{_}/{name}_bn'))

    head.add(BatchNormalization(axis = 3, name = f'{name}_head/bn'))

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

class head4():
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
            tf.keras.layers.Conv2D(256, 1, padding="same", kernel_initializer=kernel_init, name = f'conv1x1_{_}/{name}')
          )

        head.add(tf.keras.layers.ReLU())

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

class head5():
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
            tf.keras.layers.Conv2D(256, 1, padding="same", kernel_initializer=kernel_init, name = f'conv1x1_{_}/{name}')
          )

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

class Model1():
  """
  ResNet50
  """
  def __init__(self):

    # Architesture
    self.head = head(4).build_head
    self.levels = [2,3,4,5]
    self.start_level = 2
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.resnet50.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]]
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [16, 32, 64, 128]]
    self.strides        = [4,8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])

    c2_output,c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv2_block3_out","conv3_block4_out", "conv4_block6_out", "conv5_block3_out"] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')
    p2 = lateral_connection(c2_output, p3, 'p2_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)
    p2 = Conv2D(256, 3, 1, 'same', name = 'p2_conv_3x3')(p2)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p2,p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance

class Model2():
  """
  ResNet101
  """
  def __init__(self):

    # Architesture
    self.head = head(4).build_head
    self.levels = [2,3,4,5]
    self.start_level = 2
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.resnet.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]]
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [16, 32, 64, 128]]
    self.strides        = [4,8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.ResNet101(include_top=False, input_shape=[None, None, 3])

    c2_output,c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv2_block3_out","conv3_block4_out", "conv4_block6_out", "conv5_block3_out"] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')
    p2 = lateral_connection(c2_output, p3, 'p2_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)
    p2 = Conv2D(256, 3, 1, 'same', name = 'p2_conv_3x3')(p2)

    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p2,p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance

class Model3():
  """
  EfficientNetB0
  """
  def __init__(self):

    # Architesture
    self.head = head(4).build_head
    self.levels = [2,3,4,5]
    self.start_level = 2
    self.box_variance = [10, 10, 5, 5]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.efficientnet.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [16, 32, 64, 128]]
    self.strides        = [4,8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=[None, None, 3])

    c2_output,c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ['block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')
    p2 = lateral_connection(c2_output, p3, 'p2_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)
    p2 = Conv2D(256, 3, 1, 'same', name = 'p2_conv_3x3')(p2)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p2,p3,p4,p5])


  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen
  def get_box_variance(self):
    return self.box_variance

class Model4():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head(4).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance

class Model5():
  """
  VGG19
  """
  def __init__(self):

    # Architesture
    self.head = head(4).build_head
    self.levels = [2,3,4,5]
    self.start_level = 2
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.vgg19.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [16, 32, 64, 128]]
    self.strides        = [4,8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.VGG19(include_top=False, input_shape=[None, None, 3])

    c2_output,c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ['block3_conv3', 'block4_conv3', 'block5_conv3', 'block5_pool'] ]

    c5_output = BatchNormalization(axis = 3, name = 'BatchNormalization_c5')(c5_output)
    c4_output = BatchNormalization(axis = 3, name = 'BatchNormalization_c4')(c4_output)
    c3_output = BatchNormalization(axis = 3, name = 'BatchNormalization_c6')(c3_output)
    c2_output = BatchNormalization(axis = 3, name = 'BatchNormalization_c2')(c2_output)

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')
    p2 = lateral_connection(c2_output, p3, 'p2_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)
    p2 = Conv2D(256, 3, 1, 'same', name = 'p2_conv_3x3')(p2)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p2,p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance

class Model6():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head(1).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance
class Model7():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head(2).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance
  
class Model8():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head(3).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance

class Model9():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head(5).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance

class Model10():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head4(4).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance
  
class Model11():
  """
  Xception
  """
  def __init__(self):

    # Architesture
    self.head = head5(4).build_head
    self.levels = [3,4,5]
    self.start_level = 3
    self.box_variance = [0.1, 0.1, 0.2, 0.2]

    # Inputs
    self.num_classes = num_classes
    self.process_ds = process_ds(min_side, max_side, jitter, stride)
    self.input_processor = tf.keras.applications.xception.preprocess_input

    # Anchors
    self.anchors_slaces = [2 ** x for x in [0, 1 / 3, 2 / 3]] # 1.9, 1.5, 0.7
    self.aspect_ratios  = aspect_ratios
    self.areas          = [i**2 for i in [32, 64, 128]]
    self.strides        = [8,16,32]
    self.anchor_gen     = Anchor_gen.AnchorBox(self.levels, self.start_level, self.aspect_ratios, self.anchors_slaces, self.areas, self.strides)
    self.num_anchors    = len(self.aspect_ratios) * len(self.anchors_slaces)

  def __call__(self):
    return self.Model(), self.head ,self.input_processor, self.num_classes, self.num_anchors

  def Model(self):
    backbone = tf.keras.applications.Xception(include_top=False, input_shape=[None, None, 3])

    c3_output, c4_output, c5_output = [
        backbone.layers[layer_idx].output
        for layer_idx in [25, 35, 125] ]

    # FPN
    p5 = Conv2D(256, 1, 1, 'same', name = 'p5_conv_1x1')(c5_output)
    p4 = lateral_connection(c4_output, p5, 'p4_conv_1x1')
    p3 = lateral_connection(c3_output, p4, 'p3_conv_1x1')

    p5 = Conv2D(256, 3, 1, 'same', name = 'p5_conv_3x3')(p5)
    p4 = Conv2D(256, 3, 1, 'same', name = 'p4_conv_3x3')(p4)
    p3 = Conv2D(256, 3, 1, 'same', name = 'p3_conv_3x3')(p3)


    return tf.keras.Model(inputs=[backbone.inputs], outputs=[p3,p4,p5])

  def get_ds_processor(self):
    return self.process_ds

  def get_input_processor(self):
    return self.input_processor

  def get_anchor_gen(self):
    return self.anchor_gen

  def get_box_variance(self):
    return self.box_variance