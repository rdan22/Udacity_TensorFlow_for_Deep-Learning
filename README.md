# Udacity_TensorFlow_for_Deep-Learning
Udacity's Intro to TensorFlow for Deep Learning Course

## Fashion-MNIST Neural Network:

![Fashion-MNIST Network](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/Fashion-MNIST-Network.png)

The input to our models is an array of length 784. Since our neural network takes in a vector as input, these 28x28 gray-scale images are converted into a one-dimensional array of 28x28, or 784 units (flattening). The input will be fully connected to the first dense layer of our network, where we've chosen to use 128 units. 

We use **ReLU** (Rectified Linear Unit), a mathematical function that gives our dense layer more power. The ReLU function gives an output of 0 if the input is negative or zero, and if input is positive, then the output will be equal to the input. ReLU, a type of activation function, gives the network the ability to solve nonlinear problems. Finally, the output layer, our last layer, contains 10 units, because the MNIST dataset contains 10 different types of clothing. Each of these 10 output values will specify the probability that the images are that specific type of clothing( i.e., the confidence of the model). We use the softmax statement, a function that provides probabilities for each possible output class. 

We take it one step further by utilizing a **Convolutional Neural Network**. Convolutional layers can be added to the neural network model using the Conv2D layer type in Keras. This layer is similar to the Dense layer, and has weights and biases that need to be tuned to the right values. The Conv2D layer also has kernels (filters) whose values need to be tuned as well. So, in a Conv2D layer the values inside the filter matrix are the variables that get tuned in order to produce the right output.

## Going further with CNN's - Cats and Dogs:

What about datasets that contain color images? Computers interpret color images as 3-dimensional arrays, like so:

![Color Image](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/colorimage.png)

Most color images can be represented by 3 color channels: Red, Green, and Blue (RGB images). In RGB images, each color channel is represented by its own 2-dimensional array. In the case of RGB images, the depth of the 3-dimensional array is 3. 

Wouldn't it be great to use pre-trained neural networks and classify these images? This is the idea behind **Transfer learning**. Essentially, we take a neural network that has been trained on a large dataset and apply it to one it has never seen before. We only need to change the last layer of the pre-trained model, the output layer, and freeze the model. 

## Time Series Forecasting with Deep Learning:

From stock prices to weather forecasts, **time series** are everywhere. But what _is_ a time series? Time series is an ordered sequence of values usually equally spaced over time every year, day, second, etc. Time series where there is a single value at each time step are called **univariate**. Time series with multiple values at each time step are called **multivariate**. Time series analysis has many applications, particularly in forecasting, which is essentially predicting the future. 

#### Note: 

Since neural networks rely on _stochasticity_ (i.e. randomness) to initialize their parameters and gradient descent selects random batches of training data at each iteration, it's perfectly normal if the outputs you see when you run the code are slightly different each time.

### Common Patterns:

Many time series have common patterns. For example, they generally gradually drift up or down, which shows a **trend**. **Seasonality** occurs when patterns repeat at predictable intervals and particular peaks and troughs. Some time series have both trend and seasonality. On the other hand, some time series are completely unpredictable, producing **white noise**. The best we can do in this situation is identify the probability distribution and find its parameters. Combining these three results in a time series like this:

![Time Series](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/commonPatterns.png)

When you're doing time series forecasting or analysis, it helps to be able to dissect each aspect to study them separately. 

The simplest approach is to take the last value and assume the next value will be the same, i.e. **Naive Forecasting**. We generally like to measure the performance of naive forecasting. How do we do this? We typically want to split the time series into a training period, a validation period, and a testing period, called **fixed partitioning**. If the time series has some degree of seasonality, we also want to ensure that each period has a whole number of seasons. Next, we train the model on the training period and evaluate it on the validation period. After this, we train our best model one last time on the whole training + validation periods and then evaluate it on the test period. It won't always be a very reliable estimate, but it'll be reasonable. Then we have to train on the test set. This is necessary for  most time series because the most recent period is usually the one that contains the most useful information to predict the future. But giving the test data is not necessarily as reliable as it is in regular Machine Learning, it is common to just use the training and validation period.

Once we have a model and period we can evaluate the model on, we need a **metric**. The most common metric is the Mean Squared Error (MSE). The square root of the MSE, the RMSE, has the advantage that it has roughly the same scale as the values in the time series, so it's easier to interpret. Another common metric is the Mean Absolute Error (MAE), which the mean of the absolute values of the errors. If large errors are potentially dangerous and costly, then you may prefer the MSE. If the gain or loss is proportional to the size of the error, then MAE is probably better. The Mean Absolute Percentage Error (MAPE) is the mean ratio between absolute error and absolute value. 

Another simple approach is **Moving Average**. This is just the mean of the past N values. This eliminates a lot of noise, but it does not anticipate trend or seasonality, so it ends up performing worse than naive forecasting. One way to combat this is to remove the trend and seasonality from the time series. For this, a simple technique is to use **differencing**. We study the difference between the value at time <img src="https://latex.codecogs.com/gif.latex? t /> and the value at time <img src="https://latex.codecogs.com/gif.latex? t - u />, where <img src="https://latex.codecogs.com/gif.latex? u /> is the value of the time step. 
