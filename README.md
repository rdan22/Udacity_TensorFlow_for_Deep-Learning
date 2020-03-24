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

Another simple approach is **Moving Average**. This is just the mean of the past N values. This eliminates a lot of noise, but it does not anticipate trend or seasonality, so it ends up performing worse than naive forecasting. One way to combat this is to remove the trend and seasonality from the time series. For this, a simple technique is to use **differencing**. We study the difference between the value at time ![](https://latex.codecogs.com/gif.latex?t) and the value at time ![](https://latex.codecogs.com/gif.latex?t%20-%20u), where ![](https://latex.codecogs.com/gif.latex?u) is the value of a time step. This time series now has no trend nor seasonality. We get the forecasts from the differenced time series, so we need to add back the value at time ![](https://latex.codecogs.com/gif.latex?t%20-%20u). We can improve the forecasts by also removing the past noise using a moving average. 

But up to this point, we haven't talked about Machine Learning. A simple approach is to build a model that will learn to forecast the next time step given a time window before it. 

### Recurrent Neural Networks

An RNN is a neural network that contains recurrent layers, and a recurrent layer is a layer that that can sequentially process a sequence of inputs. A recurrent layer is composed of a single memory cell, which is used repeatedly to compute the outputs. A memory cell is basically a small neural network. 

![Recurrent Layer](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/recurrentlayer.png)

As shown above, there are multiple cells, but it's actually the same cell reused multiple times by the layer. At each time step, the memory cell takes the value of the input sequence at that time step, starting with ![](https://latex.codecogs.com/gif.latex?X_0) then ![](https://latex.codecogs.com/gif.latex?X_1) and so on, and it produces the output for the current time step, starting with ![](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_0) then ![](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D_1), and so on. But the memory cell also produces another output at each time step called a **state vector**, starting with ![](https://latex.codecogs.com/gif.latex?H_0) then ![](https://latex.codecogs.com/gif.latex?H_1), and so on. This state vector is fed as an additional input to the memory cell at the next time step. And hence why it is called a recurrent layer. Part of the output of the memory cell at one time step is fed back to itself at the next time step. In theory, a neural network may approximate any continuous function given it has enough neurons and can be trained properly. Similarly, if an RNN has big memory cells with a lot of neurons that can be trained properly, you can approximate any kind of algorithm. 

We will not be using the ReLU activation function here; instead we'll be using the hyperbolic tangent function. The problem with using ReLU is that RNN's have a tendency to have unstable gradients. The gradients can vanish during training or they can explode, especially when using a function like ReLU, which is non-saturated (can grow arbitrarily large). The hyperbolic tangent function is a bit more stable since it will saturate. 

#### Back Propagation Through Time:

One of the reasons it is difficult to train an RNN is that it's equivalent to training a very deep neural network with one layer per time step. During training, once the loss has been computed, back propagation computes the gradients of the loss with regards to every trainable parameter in the neural network. To do so, it propagates the gradients backwards through the RNN. TensorFlow does this by unrolling the RNN through time and treating the resulting network as an irregular feed-forward network

### Stateful vs. Stateless RNNs:

If we want an RNN to learn longer patterns, then we have two options: (1) use larger windows, and (2) we train the RNNs completely differently. 

Until now, we've trained the RNN using batches of windows sampled anywhere within the time series. For each window the RNN will run and make a prediction, but to do so, it will use an initial state of 0. Internally, it will update the state at each time step until it makes its predictions, and during training there will be a round of back propagation. After that, the RNN will drop the final state, and this is why we call it a **stateless RNN**, as at each training iteration it starts with a fresh "0" state and it drops the final state. Stateless RNNs are simple to use, but they cannot learn patterns longer than the length of the window. 

So how does a **statefull RNN** work? The batches are no longer sampled randomly. The first batch is composed of a single window at the very beginning of the time series, starting with initial state 0. After making its predictions and gradually updating the state vector, there's a round of back propagation, but this time the final state vector is not dropped. It's preserved by the RNN for the next training batch, which is composed of a single window located immediately after the previous one, starting with final state of the previous training iteration. Once we reach the end of the time series, we get a final state vector, but at this point, we can reset the state and start over at the beginning. 

Stateful RNNs are generally much less used than stateless RNNs, as back propagation does not always work and the training period will be much slower. On certain data sets, however, stateful RNNs prove to perform much better. 

### LSTM Cells

![LSTM Cell](https://github.com/rdan22/Udacity_TensorFlow_for_Deep-Learning/blob/master/lstmcell.png)

LSTM (Long Short-Term Memory) cells are much more complex than the RNN cells we have been using until now. Part of the cell is composed of a simple RNN cell (a dense layer with the hyperbolic tangent activation function). But all the other components give the cell a longer short-term memory. First, there is a state vector that gets propagated from one time step to the next. It also has a second state vector, a "long term" state vector. Notice that this goes through the cell with just two simple operations at each time step, a multiplication and an addition. Thus, the gradients can flow nicely through the cell without vanishing or exploding too much.  
