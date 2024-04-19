#image classifier cnn, 64x64 images, RGB therefore input channels = 3 (due to RGB, 1 if greyscale) making the input shape (64, 64, 3)

"""Creating a simple custom CNN from scratch without using deep learning frameworks like TensorFlow or PyTorch would involve implementing the core components of the neural 
network, including the forward pass, backpropagation, and optimization algorithm, using basic numerical computation libraries like NumPy. Here's a simplified outline of 
how you can do it:

Initialize Parameters:
Initialize the weights and biases for each layer of the network randomly or using specific initialization techniques (e.g., Xavier initialization).
Define hyperparameters such as the learning rate, number of epochs, and batch size.

Forward Pass:
Implement the forward pass through the network. This involves performing convolutions, applying activation functions, and pooling operations.
Write functions to perform convolution operations and pooling operations (e.g., max pooling).
Implement activation functions such as ReLU.

Loss Calculation:
Calculate the loss between the predicted outputs and the actual labels using an appropriate loss function (e.g., cross-entropy loss for classification).

Backpropagation:
Implement the backward pass to compute gradients of the loss function with respect to the network parameters.
Use the chain rule to propagate gradients backward through the network.
Update the weights and biases of the network using gradient descent or its variants (e.g., stochastic gradient descent, mini-batch gradient descent).

Training Loop:
Iterate over the training dataset for a specified number of epochs.
In each epoch, perform forward pass, calculate loss, perform backward pass, and update parameters using gradient descent.
Optionally, monitor training/validation loss and accuracy to track the model's performance.

Evaluation:
After training, evaluate the trained model on the test dataset to assess its performance.
Calculate metrics such as accuracy, precision, recall, and F1-score depending on the task.

Fine-tuning and Optimization:
Experiment with different network architectures, hyperparameters, and optimization techniques to improve model performance.

Deployment:
Once satisfied with the model's performance, integrate it into your desired application for inference on new data."""

# These three hyperparameters are crucial to training the neural network and need monitoring/experimentation to find optimal performance

learningRate = 0.01 # learning rate controls the rate at which weights are updated during training, larger learning rate can lead to faster convergence but 
# may cause the optimization process to become unstable and overshoot the best solution

numEpochs = 50  # an epoch is one complete pass through the entire training dataset during the training process, numEpochs determines how often the model will see the 
# entire dataset during training, too few epochs may result in underfitting and too many may result in overfitting

batchSize = 32 # Batch size is the number of training examples used in one iteration of gradient descent, Larger batch sizes can lead to faister training but 
# requires more memory and may result in less stochasticity in the gradient updates smaller sizes can lead to more stochastic gradient updates but may take longer to converge

# Convergance is the process where the optimization algorith(such as gradient descent) reaches a point where further iterations do not significantly improce the performance of the 
# model, in neural network training convergance means that the models loss function gradually decreases over time as the weights are updated during training, when a neural network 
# converges, it indicates that the model has learned the underlying patterns in the training data and is performing well on both the training and test dataset

# Stochasticity refers to the randomness or unpredictability in the optimization process, particularly in stochastic optimization algorithms like stochastic gradient descent(SGD)
# In a neural network, stochasticity arises from using mini batches of data for computing gradients instead of the entire dataset (as in batch gradient descent)
# each batch will contain a subset of training examples, leading to noisy estimates of the gradient, stochasticity introduces variability in the updates to the model parameters
# which can help the optimization process escape local minima and explore the parameter space more effectively