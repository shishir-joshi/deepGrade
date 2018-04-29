

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)




def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 100x100 pixels, and have three color channel
  input_layer = tf.reshape(features, [-1, 100, 100, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 100, 100, 3]
  # Output Tensor Shape: [batch_size, 100, 100, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 100, 100, 32]
  # Output Tensor Shape: [batch_size, 50, 50, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 50, 50, 32]
  # Output Tensor Shape: [batch_size, 50, 50, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=16,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 50, 50, 64]
  # Output Tensor Shape: [batch_size, 25, 25, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 25, 25, 64]
  # Output Tensor Shape: [batch_size, 25*25*64]
  pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 16*3*3])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 25 * 25 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  # Dropout consists in randomly setting a fraction rate of 
  # input units to 0 at each update during training time, which helps prevent overfitting
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 8]
  #layers.dense is Functional interface for the densely-connected layer
  logits = tf.layers.dense(inputs=dense, units=3) 

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  # EstimatorSpec fully defines the model to be run by an Estimator.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=3)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
		'image/encoded':tf.FixedLenFeature([], tf.string),
		'image/class/label':tf.FixedLenFeature([], tf.string),
		})

    image_record = tf.decode_raw(features['image/encoded'], tf.float32)
    image_reshape = tf.reshape(image_record, [100, 100, 3])
    image = tf.cast(image_reshape, tf.string)
    label = tf.cast(features['image/class/label'], tf.uint8)
    
    min_after_dequeue = 10
    batch_size = 100
    capacity = min_after_dequeue + 3*batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
   	[image, label], batch_size = batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch



def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features = {
		'image/encoded':tf.FixedLenFeature([], tf.string),
		'image/class/label':tf.FixedLenFeature([], tf.int64),
		})

    image_record = tf.decode_raw(features['image/encoded'], tf.uint8)
    image_uint8 = tf.reshape(image_record, [100, 100, 3])
    image = tf.cast(image_uint8, tf.float32) 
    label = features['image/class/label']
    
    min_after_dequeue = 10
    batch_size = 100
    capacity = min_after_dequeue + 3*batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
   	[image, label], batch_size = batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch




# input pipeline
def input_pipeline(filenames, num_epochs=None):
  filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
  image, label = read_and_decode(filename_queue)
  return image, label
  
#  filename_queue = tf.train.string_input_producer(filenames, num_epochs=10, shuffle=True)   
  
filenamet = ["./outputs/train-00000-of-00002"]
filenamev = ["./outputs/validation-00000-of-00002"]
def train_input_fn():
  # TFRecordDataset opens a protobuf and reads entries line by line
  # could also be [list, of, filenames]
  dataset = tf.data.TFRecordDataset(filenamet)

  # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(decode)
#  dataset = dataset.batch(30)
  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()

  return features, labels
 
def eval_input_fn():
  dataset = tf.data.TFRecordDataset(filenamev)
   # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(decode)
#  dataset = dataset.batch(3)
  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()

  return features, labels


def main(unused_argv):
  # Load training and eval data
# train_data = tf.train.string_input_producer(["./output/train-00000-of-00002.tfrecord"], num_epochs=10)
# train_labels = np.asarray("./labels.txt", dtype=np.int32)
# eval_data = tf.train.string_input_producer(["./output/validation-00000-of-00002.tfrecord"], num_epochs=10)  # Returns np.array
# eval_labels = np.asarray("./labels.txt", dtype=np.int32)

  train_data , train_labels = input_pipeline(["./outputs/train-00000-of-00002"], num_epochs = 10)
  eval_data , eval_labels = input_pipeline(["./outputs/validation-00000-of-00002"], num_epochs = 1)
    
      
  # Create the Estimator
  # The Estimator object wraps a model which is specified by a model_fn, which, 
  # given inputs and a number of other parameters, returns the ops necessary 
  # to perform training, evaluation, or predictions.
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)


  classifier.train(
      input_fn=train_input_fn
      )

      
         
  eval_results = classifier.evaluate(
      input_fn=eval_input_fn
      )
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
