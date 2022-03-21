import sys
import tqdm_utils
import download_utils

download_utils.link_all_keras_resources()

from __future__ import print_function
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
%matplotlib inline

#define the ReLu layer

class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(input, np.zeros(input.shape))
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad      
      
#Define the dense layer. We have some kinds of optimizer and weights inicialization, you can try test these possibles 
#configurations to analyse the results and get some conclusions about their properties.
      
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.05, init_method="default", l2=0, optimizer="default",
                 alpha=0.9):
        
        super().__init__()
        self.learning_rate = learning_rate
        
        if init_method == "default":
        # initialize weights with small random numbers. We use normal initialization, 
        # but surely there is something better. Try this once you got it working: http://bit.ly/2vTlmaJ
            self.weights = np.random.randn(input_units, output_units)*0.01
            
        elif init_method == "xavier":
            self.weights = norm.rvs(loc=0, scale=1/output_units, size= input_units*output_units).reshape(input_units, output_units)
            
        elif init_method == "kaiming":
            self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / input_units)
            
        self.l2 = l2
        self.optimizer = optimizer
        self.alpha = alpha
        self.nu_weights = 0
        self.nu_biases = 0
        self.biases = np.zeros(output_units)
            
    def forward(self,input):
        return np.dot(input, self.weights) + self.biases
    
    def backward(self,input,grad_output):
        
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T) #df/dx = df/ds * W.T and here df/ds = grad_output
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output) 
        grad_biases = np.sum(grad_output, axis = 0) 
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # Here we perform a stochastic gradient descent step. 

        if self.optimizer == "default":
            self.weights = self.weights - self.learning_rate * grad_weights
            self.biases = self.biases - self.learning_rate * grad_biases
            
        elif self.optimizer == "momentum":
            self.nu_weights = self.alpha * self.nu_weights + self.learning_rate * grad_weights
            self.weights = self.weights - self.nu_weights

            self.nu_biases = self.alpha * self.nu_biases + self.learning_rate * grad_biases
            self.biases = self.biases - self.nu_biases
            
        def sum_squared_weights(self):
        # compute sum of the squared weights of the layer for regularized loss function 
            return self.l2 * (self.weights ** 2).sum()
        
        return grad_input 
      
def softmax_crossentropy_with_logits(logits,reference_answers):
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
   
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]
  

def create_network():
  network = []
  network.append(Dense(X_train.shape[1],100))
  network.append(ReLU())
  network.append(Dense(100,200))
  network.append(ReLU())
  network.append(Dense(200,10))
  return network


def forward(network, X):
    activations = []
    input = X
    
    # <your code here>
    for layer in network:
        output = layer.forward(input)
        activations.append(output)
        input = output
        
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)

def train(network,X,y):
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    
    # propagate gradients through the network>
    grad_output = loss_grad
    layer_inputs = layer_inputs[:-1]
    for input,layer in zip(layer_inputs[::-1],network[::-1]):
        grad_output = layer.backward(input,grad_output)
        
    return np.mean(loss)

# We split data into minibatches, feed each such minibatch into the network and update weights.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in tqdm_utils.tqdm_notebook_failsafe(range(0, len(inputs) - batchsize + 1, batchsize)):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
        
from preprocessed_mnist import load_dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

plt.figure(figsize=[6,6])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title("Label: %i"%y_train[i])
    plt.imshow(X_train[i].reshape([28,28]),cmap='gray');
    
#now create the model and train it:

network = create_network()

for epoch in range(25):

    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=32,shuffle=True):
        train(network,x_batch,y_batch)
    
    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))
    
    clear_output()
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])
    plt.plot(train_log,label='train accuracy')
    plt.plot(val_log,label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
 
