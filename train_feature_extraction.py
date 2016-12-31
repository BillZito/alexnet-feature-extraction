import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

batch_size = 16
img_size = 32
depth = 3
num_classes = 43
# load training data with pickle
with open("train.p", "rb") as f:
  data = pickle.load(f)
# print('data is', data)

#load names

# Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(data["features"], data["labels"], test_size=0.10, random_state=0)
# print('x trainshape', X_train.shape)
# print('ys are', y_train, y_test)

# Define placeholders and resize operation.
x_placeholder = tf.placeholder(tf.float32, shape=(None, img_size, img_size, depth))
y_placeholder = tf.placeholder(tf.int32, shape=(None))
resized = tf.image.resize_images(x_placeholder, (227, 227))
# print('x placeholder', x_placeholder)
print("resized", resized)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
#fc7 is 4096 by 128
shape = (fc7.get_shape().as_list()[-1], num_classes)
print('fc7 shape', shape)

# Add the final layer for traffic sign classification.
fc8w = tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.1))
fc8b = tf.Variable(tf.zeros(shape[1]))
fc8 = tf.matmul(fc7, fc8w) + fc8b

# TODO: Define loss, training, accuracy operations.

epochs = 30
rate = .001
#loss is stochastic gradient descent-- or is it atomizer? 
#takes predictions, actual values, and calculates cross_entropy (distance)
#cross_entropy is just another word for loss? 
#logits are the result of the tf operation
logits = fc8
#one_hot are the potential answers
one_hot = tf.one_hot(y_placeholder, 43)
#cross-entropy is the distance of each prediction softmaxed and then calculated
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot)
#loss should minimize cross_entropy? (which is distance from each point?)
loss_operation = tf.reduce_mean(cross_entropy)

#optimizer set with our learning rate--how does adam work?
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#the training operation should minimize loss
training_operation = optimizer.minimize(loss_operation)

# correct prediction is setting max val of 1st axis aka columns (which are 0 or 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot, 1))
#accuracy operation-- how close we are to answer-- casting score out of 1 and trying to reduce mean?
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#evaluate--for each sample, see if correct and divide by 100
def evaluate(X_data, y_data):
  num_examples = len(X_train)
  sess = tf.get_default_session()
  total_accuracy = 0

  #run it for each 128 examples in set-- can do it for subset 
  for offset in range(0, num_examples, batch_size):
    batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
    accuracy = sess.run(accuracy_operation, feed_dict={x_placeholder:batch_x, y_placeholder:batch_y})
    total_accuracy += (accuracy * len(batch_x))

  return total_accuracy/ num_examples
# training involves running the tf.session

#why don't we need tf.graph()? 
# with tf.graph() as graph:
  # run model -- put fc7 and fc8 here then return it

# with tf.session as sess
with tf.Session() as sess: 
  
  sess.run(tf.global_variables_initializer())
  num_examples = 100
  print('num examples', num_examples)
  print('batch_size', batch_size)

  print("training")

  for i in range(epochs):
    X_train, y_train = shuffle(X_train[0:num_examples], y_train[0:num_examples])
    print("epoch", i + 1)
    for offset in range(0, num_examples, batch_size):
      print('offset', offset)
      # batch_num = offset // num_examples
      # print("batch %d" % batch_num)
      batch_x, batch_y = X_train[offset: offset + batch_size], y_train[offset: offset + batch_size]
      sess.run(training_operation, feed_dict={x_placeholder:batch_x, y_placeholder:batch_y})
      
      #if we had a validation set, we"d do it here
    test_accuracy = evaluate(X_test[0:100], y_test[0:100])
    print("val accuracy:", test_accuracy)

  # try:
  #       saver
  #   except NameError:
  #       saver = tf.train.Saver()
  #   saver.save(sess, 'lenet')
  #   print("Model saved")




# TODO: Train and evaluate the feature extraction model.
