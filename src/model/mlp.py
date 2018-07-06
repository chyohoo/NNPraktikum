
import numpy as np

from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', inputActivation = "sigmoid", outputActivation='softmax',
                 loss='ce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        # self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if loss == 'ce':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = [] * 10 

        self.inputActivation = inputActivation
        # Input layer
        self.layers.append(LogisticLayer(train.input.shape[1], 10,
                           None, inputActivation, False))

        # @Author  : Haoye
        # Hidden layer

        for i in range(1,8):
            self.layers.append(LogisticLayer(10,10,None,inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(10, 10,
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters 
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        # @Author  : Yingzhi ,Haoye
        # do forward pass for all layers
        self.layers[0].forward(inp)
        
        for i in range(1,len(self.layers)):
            self.layers[i].inp = np.insert(self.layers[i - 1].outp,0,1) # add bias values ("1"s) at the beginning
            self.layers[i].forward(self.layers[i].inp)
        # return inp

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # @Author  : Yingzhi
        for i, layer in enumerate(reversed(self.layers)):
            if layer.isClassifierLayer:
                next_derivatives = - self.loss.calculateDerivative(target, self.layers[i].outp)
                next_weights = np.ones(layer.shape[1])

            layer.computeDerivative(next_derivatives, next_weights)
            next_derivatives = layer.deltas
            next_weights = layer.weights[1:]


    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        # @Author : Yue Ning
        # in order to realize this update_weights function, need deltas
        # to get deltas in train function
        for n, layer in enumerate(self.layers):
            for neuron in range(0, layer.nOut):
                layer.weights[:, neuron] -= (learningRate *
                                                layer.deltas[neuron] *
                                                layer.inp)
            self.layers[n].weights = layer.weights

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # @Author  : Haoye
        for epoch in range(self.epochs):
            if verbose:
                print("Training epcho {0}/{1}.."
                      .format(epoch + 1, self.epochs))

                for img, label in zip(self.trainingSet.input,
                                      self.trainingSet.label):
                    self._feed_forward(img)

                    self._compute_error(label)
                    self._update_weights(self.learningRate)


            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                       .format(accuracy * 100))
                print("---------------------------------")



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        # Author: Yue Ning
        outp = np.argmax(self._feed_forward(test_instance))
        return outp


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
