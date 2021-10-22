import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(
            0.0, self.input_nodes ** -0.5, (self.input_nodes, self.hidden_nodes)
        )

        self.weights_hidden_to_output = np.random.normal(
            0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.output_nodes)
        )
        self.lr = learning_rate

        #### TODO (Done): Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # Learn more about lambda here => https://www.w3schools.com/python/python_lambda.asp
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        # def sigmoid(x):
        #    return 1 / (1 + np.exp(-x))
        # self.activation_function = sigmoid

    def train(self, features, targets):
        """Train the network on batch of features and targets.

        Arguments
        ---------

        features: 2D array, each row is one data record, each column is a feature
        targets: 1D array of target values

        """
        # Refrence to Shape of arrays: https://www.python-course.eu/numpy_create_arrays.php
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(
                X
            )  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs,
                hidden_outputs,
                X,
                y,
                delta_weights_i_h,
                delta_weights_h_o,
            )
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        """Implement forward pass here

        Arguments
        ---------
        X: features batch

        """
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO(Done): Hidden layer - Replace these values with your calculations.
        # Learn more about "numpy.dot" here: https://numpy.org/doc/stable/reference/generated/numpy.dot.html

        hidden_inputs = np.dot(
            X, self.weights_input_to_hidden
        )  # signals into hidden layer
        hidden_outputs = self.activation_function(
            hidden_inputs
        )  # signals from hidden layer

        # TODO(Done): Output layer - Replace these values with your calculations.
        final_inputs = np.dot(
            hidden_outputs, self.weights_hidden_to_output
        )  # signals into final output layer
        # final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from the final output layer

        return final_outputs, hidden_outputs
        # print(final_outputs, hidden_outputs)

    def backpropagation(
        self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o
    ):
        """Implement backpropagation

        Arguments
        ---------
        final_outputs: output from forward pass
        y: target (i.e. label) batch
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers

        """
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO (Done): Output error - Replace this value with your calculations.
        error = (
            y - final_outputs
        )  # Output layer error is the difference between desired target and actual output.

        # TODO(Done): Backpropagated error terms - Replace these values with your calculations.
        # https://ryanwingate.com/intro-to-machine-learning/deep/backpropagation-implementations/
        # output_error_term = error * final_outputs * (1 - final_outputs)
        output_error_term = error
        # print(output_error_term)

        # TODO (Done): Calculate the hidden layer's contribution to the error
        # Refrence to more understanding: Chapter 2. Neural Networks - Lesson 2 Gradient descent
        # hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        hidden_error = np.dot(
            # A mentor gave me a hint for this part https://knowledge.udacity.com/questions/724105
            self.weights_hidden_to_output,
            error,
        )  # After passing error as a param, the results changed
        # Explained here:
        # https://julienbeaulieu.gitbook.io/wiki/sciences/machine-learning/neural-networks/backpropagation
        # hidden_error = error * self.weights_hidden_to_output
        # hidden_error_term = hidden_error.T * (hidden_outputs * (1 - hidden_outputs))
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # TODO(Done): Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)
        # Refrence to np.matmul method: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # delta_weights_i_h += np.matmul(X[:, None], hidden_error_term)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        delta_weights_h_o += hidden_outputs.reshape(-1, 1) * output_error_term
        # delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """
        **Update weights on gradient descent step**

        Arguments
        ---------
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        n_records: number of records

        """

        # TODO (Done): Update the weights with gradient descent step
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        """Run a forward pass through the network with input features

        Arguments
        ---------
        features: 1D array of feature values
        """

        #### Implement the forward pass here ####
        # TODO (Done): Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(
            features, self.weights_input_to_hidden
        )  # signals into hidden layer

        hidden_outputs = self.activation_function(
            hidden_inputs
        )  # signals from the hidden layer

        # TODO (Done): Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(
            hidden_outputs, self.weights_hidden_to_output
        )  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer (no activation function, f(x)=x

        return final_outputs


#########################################################
# Set your hyperparameters here ()
##########################################################
iterations = 5000
learning_rate = 0.5
hidden_nodes = 30
output_nodes = 1
# Results => Progress: 100.0% ... Training loss: 0.072 ... Validation loss: 0.167

# hyperparameters I tried earliear and thier results: (WARNING: When I changed the hidden_error variable, they all changed )
# [1]
# iterations = 3000
# learning_rate = 0.5
# hidden_nodes = 25
# output_nodes = 1
# Results => Progress: 100.0% ... Training loss: 0.643 ... Validation loss: 0.862


# [2]
# iterations = 3500 # 4000 #5000
# learning_rate = 0.5 # 0.7 #0.9
# hidden_nodes = 20
# output_nodes = 1
# Results => Progress: 100.0% ... Training loss: nan ... Validation loss: nan (IDK why but prediction doen't happen ! )
# RuntimeWarning: invalid value encountered in multiply delta_weights_h_o += hidden_outputs.reshape(-1, 1) * output_error_term


# [3]
# iterations = 5000
# learning_rate = 0.5
# hidden_nodes = 10
# output_nodes = 1
# Results => Progress: 0.1% ... Training loss: 2.269 ... Validation loss: 2.318

# [4]
# iterations = 2500
# learning_rate = 0.5
# hidden_nodes = 10 #15
# output_nodes = 1
# Results => Progress: 100.0% ... Training loss: nan ... Validation loss: nan (IDK why but prediction doen't happen ! )
# RuntimeWarning: invalid value encountered in multiply delta_weights_h_o += hidden_outputs.reshape(-1, 1) * output_error_term
# (╥﹏╥) (͠◉_◉᷅ )


# [5]
# iterations = 2500
# learning_rate = 0.5
# hidden_nodes = 30
# output_nodes = 1
# Results => Progress: 100.0% ... Training loss: 0.750 ... Validation loss: 0.895

# [6]
# iterations = 3500
# learning_rate = 0.5
# hidden_nodes = 30
# output_nodes = 1
# Results => Progress: 100.0% ... Training loss: 0.740 ... Validation loss: 0.921

# [7]
# iterations = 4000
# learning_rate = 0.5
# hidden_nodes = 30
# output_nodes = 1
# Results => Progress: 100.0% ... Training loss: 0.678 ... Validation loss: 0.880


# [8]
# iterations = 4000
# learning_rate = 0.5
# hidden_nodes = 7
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.622 ... Validation loss: 0.836

# [9]
# iterations = 5000
# learning_rate = 0.6
# hidden_nodes = 10
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.613 ... Validation loss: 0.822

# [10]
# iterations = 5000
# learning_rate = 0.8
# hidden_nodes = 10
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.611 ... Validation loss: 0.802

# [11]
# iterations = 5000
# learning_rate = 0.8
# hidden_nodes = 15
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.614 ... Validation loss: 0.782

# [12]
# iterations = 5000
# learning_rate = 0.9
# hidden_nodes = 15
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.617 ... Validation loss: 0.789

# [13]
# iterations = 8000
# learning_rate = 0.5
# hidden_nodes = 30
# output_nodes = 1
# Progress: 100.0% ... Training loss: 0.647 ... Validation loss: 0.836
# (ㆆ_ㆆ)