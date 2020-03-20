#A neural network to identify items of clothing in images
#feature = input image, label = correct output that specifies the piece of clothing the image depicts
#dataset = Fashion-MNIST, 70,000 28x28 pixel gray-scale images of clothing 
#We use 60,000 to train (85.7%) and 10,000 to test (14.3%) 
#Run pip install -U tensorflow_datasets to install the dataset 

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#access the Fashion MNIST directly from TensorFlow, using the Datasets API
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#each image is mapped to a single label.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

#Preprocess the data
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels
#The map function applies the normalize function to each element in the train
#and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)
#The first time you use the dataset, the images will be loaded from disk
#Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

#Look at the processed data
#Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

#Plot the image 
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#Display the first 25 images from the training set and display the class name below each image. 
#Verify that the data is in the correct format and we're ready to build and train the network.
plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

#We must configure the layers of the model, then compile it
#Configuration
model = tf.keras.Sequential([
	#This layer transforms the images from a 2d-array of 28x28 pixels, to a 1d-array of 784 pixels (28*28). 
	#Think of this layer as unstacking rows of pixels in the image and lining them up. 
	#This layer has no parameters to learn, as it only reformats the data.
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #A densely connected layer of 128 neurons. Each neuron (or node) takes input from all 784 nodes in the previous layer, 
    #weighting that input according to hidden parameters which will be learned during training, and outputs a single value to the next layer.
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    #A 10-node softmax layer, with each node representing a class of clothing. As in the previous layer, each node takes input from the 128 nodes in the layer before it. 
    #Each node weights the input according to learned parameters, and then outputs a value in the range [0, 1], representing the probability that the image belongs to that class. 
    #The sum of all 10 node values is 1.
    tf.keras.layers.Dense(10)
])
#Compilation
#Optimizer -An algorithm for adjusting the inner parameters of the model in order to minimize loss.
#Loss function-An algorithm for measuring how far the model's outputs are from the desired output. The goal of training is this measures loss.
#Metrics -Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Training the model
#Repeat forever by specifying dataset.repeat() (the epochs parameter described below limits how long we perform training).
#The dataset.shuffle(60000) randomizes the order so our model cannot learn anything from the order of the examples.
#And dataset.batch(32) tells model.fit to use batches of 32 images and labels when updating the model variables.
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)
#Feed the training data to the model using train_dataset.
#The model learns to associate images and labels.
#The epochs=5 parameter limits training to 5 full iterations of the training dataset, so a total of 5 * 60000 = 300000 examples.
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

#Accuracy: we have to compare how our model performs on the test data set
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

#make predictions
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)
predictions.shape

#full set of 10 class predictions
def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Let's plot several images with their predictions. 
#Correct prediction labels are blue and incorrect prediction labels are red. 
#The number gives the percent (out of 100) for the predicted label. 
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
#Grab an image from the test dataset
img = test_images[0]
print(img.shape)

#tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:
img = np.array([img])
print(img.shape)

#now predict the image
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#model.predict returns a list of lists, one for each image in the batch of data. Grab the predictions for our (only) image in the batch:
np.argmax(predictions_single[0])
