import numpy as np
import sys
import os
import random
import cv2
import math
from matplotlib import pyplot as plt
from sklearn import preprocessing

def getImage(name):
    return cv2.imread(name, cv2.IMREAD_COLOR)

def getImageGray(name):
    return cv2.imread(name, cv2.IMREAD_GRAYSCALE)

def getBlackImage(width, height):
    img = getImageGray("Lenna.png")
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[i][j] = 0
    return cv2.resize(img, (width,height))

def writeImage(name, img):
    cv2.imwrite(name, img)

## Define a sigmoid function.
def sigmoid(x):
    return 1/(1+np.exp(-x))

## Derivative of the sigmoid function.
## Derivative = sigmoid(x) * sigmoid(1-x)
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

# alternative activation function
def ReLU(x):
    return np.maximum(0.0, x)
# derivation of relu
def ReLU_derivation(x):
    if x <= 0:
        return 0
    else:
        return

class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        mu, sigma = -1, 10
        self.weights_in_hidden = np.random.normal(mu, sigma, size=(self.no_of_in_nodes,
                                       self.no_of_hidden_nodes))

        self.weights_hidden_out = np.random.normal(mu, sigma, size=(self.no_of_hidden_nodes,
                                        self.no_of_out_nodes))

    def feed_forward(self, input_vector):
        output_vector1 = np.dot(input_vector, self.weights_in_hidden)
        output_vector_hidden = sigmoid(output_vector1)
        #output_vector_hidden = output_vector1
        self.hidden_layer = output_vector_hidden

        output_vector2 = np.dot(output_vector_hidden, self.weights_hidden_out)
        output_vector_network = sigmoid(output_vector2)
        #output_vector_network = output_vector2
        self.predicts = output_vector_network

    def loss_function(self, input_array, target_array):
        self.feed_forward(input_array)
        self.loss = np.mean(np.square(self.predicts - target_array))

    def back_propagation(self, input_vector, target):
        d_weight_2 = np.dot(self.hidden_layer.T, 2*(self.predicts - target) *  self.predicts * (1 - self.predicts))

        d_weight_1 = np.dot(input_vector.T,  ( self.weights_hidden_out.T *2*(self.predicts - target)  * self.hidden_layer * (1 - self.hidden_layer)))
        self.d_weight_2 = d_weight_2
        self.d_weight_1 = d_weight_1

    def train(self, input_vector, target_vector):

        self.feed_forward(input_vector)
        self.back_propagation(input_vector, target_vector)

        self.weights_hidden_out -= self.learning_rate * self.d_weight_2
        self.weights_in_hidden  -= self.learning_rate * self.d_weight_1


def main():

    train_x = []
    test_x = []

    train_y = []
    test_y = []

    for i in range(800):
        one = random.randint(0,256) / 256.0
        two = random.randint(0,256) / 256.0
        three = random.randint(0,256) / 256.0

        train_x.append([one, two, three])
        train_y.append([((one + two + three) / 3)])

    for i in range(200):
        one = random.randint(0,256) / 256.0
        two = random.randint(0,256) / 256.0
        three = random.randint(0,256) / 256.0

        test_x.append([one, two, three])
        test_y.append([(int)((one + two + three) / 3)])

    simple_network = NeuralNetwork(no_of_in_nodes=3,
                                   no_of_out_nodes=1,
                                   no_of_hidden_nodes=100,
                                   learning_rate=0.6)

    for epoch in range(500):
        for i in range(len(train_x)):
            input_vector = train_x[i]
            #input_vector = train_x[i,:]  # shape = (3, )
            input_vector = np.expand_dims(input_vector, axis=0)  # shape = (1, 3)
            target_vector = train_y[i]

            simple_network.train(input_vector, target_vector)
            simple_network.loss_function(input_vector, target_vector)
            #print(simple_network.d_weight_2)
            #print(simple_network.predicts)
            #print(simple_network.loss)
            #print("-----------------")

        simple_network.loss_function(train_x, train_y)
        # print the loss on training dataset for the whole dataset.
        print("Training loss:" + str(simple_network.loss ))

        # evalue the model on testing dataset for each 10 epochs.
        if epoch % 10 == 0:
            simple_network.loss_function(test_x, test_y)
            print("Testing loss: " + str(simple_network.loss))

    # Time to work with the image
    nameStr = input("Enter the name of your image: ")
    img = getImage(nameStr)
    imgGray = getBlackImage(len(img), len(img[0]))

    writeImage("Black.png", imgGray)

    for x in range(len(img)):
        for y in range(len(img[x])):
            #print(img[x][y])
            simple_network.feed_forward(img[x][y])
            #print(simple_network.predicts)
            imgGray[x][y] = (int)(simple_network.predicts * 256.0)

    writeImage("Output.png", imgGray)

main()
