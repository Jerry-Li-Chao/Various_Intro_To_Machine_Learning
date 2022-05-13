import numpy as np

import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x,self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        run_result = self.run(x)
        binary_outcome = nn.as_scalar(run_result)
        # all positive result "collapse" to 1, all negative result "collapse" to -1
        if binary_outcome < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        keep_updating = True

        while keep_updating:
            # assume we will not predict wrong, if wrong then correct this flag
            keep_updating = False
            for x, y in dataset.iterate_once(1):
                true_y = nn.as_scalar(y)
                if self.get_prediction(x) != true_y:
                    # if it predicted wrong, we need to update the weights
                    # we need to keep updating to check if the new weight is going to predict correct y
                    keep_updating = True
                    nn.Parameter.update(self.w, x, true_y)
                    


class LogisticRegressionModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Logistic Regression instance.

        A perceptron classifies data points as either belonging to a particular
        class (1) or not (0). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        "*** Your Code Here ****"
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the model.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the logistic regression to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the predicted probability)
        """
        "*** YOUR CODE HERE ***"
        return nn.Sigmoid(nn.DotProduct(x, self.w))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or 0
        """
        "*** YOUR CODE HERE ***"
        run_result = self.run(x)
        predicted_probability = nn.as_scalar(run_result)
        # we split the probability in half for the binary outcomes
        if predicted_probability < 0.5:
            return 0
        else:
            return 1

    def calculate_log_likelihood_gradient(self, x, y):
        """
        Calculate the maximum likelihood gradient for a single datapoint x and y

        nn.Subtract and nn.ScalarMatrixMultiply might be helpful to use
        Returns: a Node representing the gradient of the maximum log-likelihood of the data with respect to the params
        """
        # using the formula on the bottom of page 5 of the writeup
        return nn.ScalarMatrixMultiply(nn.Subtract(y, self.run(x)), x)

    def train(self, dataset, learning_rate, iterations):
        """
        Train the logistic regression model using stochastic gradient ascent (a single datapoint at a time).

        Use dataset.pick_random(batch_size) to sample random (x, y) data pairs from the dataset

        Use the update function on your parameter to make further changes.
        """
        "*** Your Code Here ***"
        for iter in range(iterations):
            # pick a random data pair and update weight beased on log likelihood
            x, y = dataset.pick_random(1)
            self.w.update(self.calculate_log_likelihood_gradient(x, y), learning_rate)


class ClassificationModel(object):
    """
    A generic neural network model

    The model should be initialized using the provided input hyperparameters.
    """
    def __init__(self, input_dim=784, hidden_dim=4, layers=2, output_dim=10):
        """Initialize your model parameters here"""
        self.weights = []
        self.biases = []
        "*** YOUR CODE HERE ***"
        self.layer_count = layers
        self.weights.append(nn.Parameter(input_dim, hidden_dim))
        self.biases.append(nn.Parameter(1, hidden_dim))
        # Hard coded 4 here to represent 4 entries of weight and biases
        for x in range(4-2):
            self.weights.append(nn.Parameter(hidden_dim, hidden_dim))
            self.biases.append(nn.Parameter(1, hidden_dim))
        self.weights.append(nn.Parameter(hidden_dim, output_dim))
        self.biases.append(nn.Parameter(1, output_dim))
        
        self.params = self.weights + self.biases

    def get_params(self):
        """Should return a list of all the parameters"""
        return self.params

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        layer = nn.Linear(x, self.weights[0])
        layer_final = nn.ReLU(nn.AddBias(layer, self.biases[0]))
        # Hard coded 4 here to represent 4 entries of weight and biases
        for i in range(4-2):
            layer = nn.Linear(layer, self.weights[i+1])
            layer_final = nn.ReLU(nn.AddBias(layer, self.biases[i+1]))

        output = nn.Linear(layer_final, self.weights[4-1])

        return nn.AddBias(output, self.biases[4-1])

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset, learning_rate, epochs, batch_size):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for e in range(epochs):
            for x, y in dataset.iterate_once(batch_size):
                single_gradient = nn.gradients(self.get_loss(x, y), self.params)
                for each in range(len(self.params)):
                    self.params[each].update(single_gradient[each], learning_rate)

    def eval(self, eval_dataset):
        """
        Runs evaluation using the accuracy metric. You do not need to implement this part but should take a look.
        """
        pred_label = None
        true_label = None
        for x,y in eval_dataset.iterate_once(batch_size=len(eval_dataset)):
            pred_label = np.argmax(self.run(x).data, axis=1)
            true_label = np.argmax(y.data, axis=1)

        return (pred_label == true_label).mean()

    def get_prediction(self, x):
        """
        Gets a list of predictions for a databatch x
        """
        return np.argmax(self.run(x).data, axis=1)


