# Udacity_TensorFlow_for_Deep-Learning
Code for Udacity's Intro to TensorFlow for Deep Learning

Fashion-MNIST Neural Network:

![Fashion-MNIST Network](Fashion-MNIST\/Network.png)

The input to our models is an array of length 784. Since our neural network takes in a vector as input, these 28x28 images are converted into a one-dimensional array of 28x28, or 784 units (flattening). 
The input will be fully connected to the first dense layer of our network, where we've chosen to use 128 units. We use ReLU (Rectified Linear Unit), a mathematical function that gives our dense layer more power. 
The ReLU function gives an output of 0 if the input is negative or zero, and if input is positive, then the output will be equal to the input. ReLU, a type of activation function, gives the network the ability to solve nonlinear problems. 
Finally, the output layer, our last layer, contains 10 units, because the MNIST dataset contains 10 different types of clothing. Each of these 10 output values will specify the probability that the images are that specific type of clothing( i.e., the confidence of the model). 
We use the softmax statement, a function that provides probabilities for each possible output class. 
