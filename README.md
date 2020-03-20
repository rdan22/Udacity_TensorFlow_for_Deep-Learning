# Udacity_TensorFlow_for_Deep-Learning
Udacity's Intro to TensorFlow for Deep Learning Course

## Fashion-MNIST Neural Network:

![Fashion-MNIST Network](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/Fashion-MNIST-Network.png)

The input to our models is an array of length 784. Since our neural network takes in a vector as input, these 28x28 gray-scale images are converted into a one-dimensional array of 28x28, or 784 units (flattening). 
The input will be fully connected to the first dense layer of our network, where we've chosen to use 128 units. We use **ReLU** (Rectified Linear Unit), a mathematical function that gives our dense layer more power. 
The ReLU function gives an output of 0 if the input is negative or zero, and if input is positive, then the output will be equal to the input. ReLU, a type of activation function, gives the network the ability to solve nonlinear problems. 
Finally, the output layer, our last layer, contains 10 units, because the MNIST dataset contains 10 different types of clothing. Each of these 10 output values will specify the probability that the images are that specific type of clothing( i.e., the confidence of the model). We use the softmax statement, a function that provides probabilities for each possible output class. 

We take it one step further by utilizing a **Convolutional Neural Network**. Convolutional layers can be added to the neural network model using the Conv2D layer type in Keras. This layer is similar to the Dense layer, and has weights and biases that need to be tuned to the right values. The Conv2D layer also has kernels (filters) whose values need to be tuned as well. So, in a Conv2D layer the values inside the filter matrix are the variables that get tuned in order to produce the right output.

## Going further with CNN's - Cats and Dogs:

What about datasets that contain color images? Computers interpret color images as 3-dimensional arrays, like so:

![Color Image](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/colorimage.png)

Most color images can be represented by 3 color channels: Red, Green, and Blue (RGB images). In RGB images, each color channel is represented by its own 2-dimensional array. In the case of RGB images, the depth of the 3-dimensional array is 3. 

Wouldn't it be great to use pre-trained neural networks and classify these images? This is the idea behind **Transfer learning**. Essentially, we take a neural network that has been trained on a large dataset and apply it to one it has never seen before. We only need to change the last layer of the pre-trained model, the output layer, and freeze the model. 

## Time Series Forecasting with Deep Learning:

From stock prices to weather forecasts, **time series** are everywhere. But what _is_ a time series? Time series is an ordered sequence of values usually equally spaced over time every year, day, second, etc. Time series where there is a single value at each time step are called **univariate**. Time series with multiple values at each time step are called **multivariate**. Time series analysis has many applications, particularly in forecasting, which is essentially predicting the future. 

### Common Patterns:

Many time series have common patterns. For example, they generally gradually drift up or down, which shows a **trend**. **Seasonality** occurs when patterns repeat at predictable intervals and particular peaks and troughs. Some time series have both trend and seasonality. On the other hand, some time series are completely unpredictable, producing **white noise**. The best we can do in this situation is identify the probability distribution and find its parameters. This is a good start. 
