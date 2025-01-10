import nn
import numpy as np

unit_one, unit_neural_num = 1, 4

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
        weights = self.get_weights()
        result = nn.DotProduct(weights, x)
        return result

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return -unit_one if nn.as_scalar(self.run(x)) < 0 else unit_one

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = True
            for x, y in dataset.iterate_once(unit_one):
                if self.get_prediction(x) != (label := nn.as_scalar(y)):
                    self.get_weights().update(x, label)
                    flag = False
            if flag:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        neural_num = unit_neural_num ** 3
        self.w1, self.b1 = nn.Parameter(1, neural_num), nn.Parameter(1, neural_num)
        self.w2, self.b2 = nn.Parameter(neural_num, 1), nn.Parameter(1, 1)
        self.activate1, self.loss_function = nn.ReLU, nn.SquareLoss

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        f_x1 = self.activate1(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        f_x2 =  nn.AddBias(nn.Linear(f_x1, self.w2), self.b2)
        return f_x2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pre_out = self.run(x)
        loss = self.loss_function(pre_out, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size, flagc, start_lr, end_lr, delta_loss = (
            20 * unit_one, False, 6e-2, 1e-2, (1 / 20))
        gamma = (end_lr / start_lr) ** delta_loss
        iter, lr = unit_one, start_lr

        while not flagc:
            for x, y in dataset.iterate_once(batch_size):
                loss_num = nn.as_scalar(self.get_loss(x, y))
                if loss_num * 10 - delta_loss < 0:
                    flagc = True
                    break
                grads = nn.gradients(self.get_loss(x, y),
                        [self.w1, self.b1, self.w2, self.b2])
                for param, grad in zip([self.w1, self.b1, self.w2, self.b2], grads):
                    param.update(grad, -lr)
                lr = lr if iter > 2 else lr * gamma
            iter = iter + 1

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.x1 = nn.Parameter(784, 250)
        self.x2 = nn.Parameter(250, 10)
        self.y1 = nn.Parameter(1, 250)
        self.y2 = nn.Parameter(1, 10)

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
        linear1 = nn.Linear(x, self.x1)
        bias1 = nn.AddBias(linear1, self.y1)
        relu1 = nn.ReLU(bias1)
        linear3 = nn.Linear(relu1, self.x2)
        predicted = nn.AddBias(linear3, self.y2)
        return predicted

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

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate, acc, batch_size = 0.5, 0, 100

        while acc < 0.975:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, [self.x1, self.y1, self.x2, self.y2])
                for var, grad in zip([self.x1, self.y1, self.x2, self.y2], grads):
                    var.update(grad, -learning_rate)
                acc = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(47, 512)
        self.w2 = nn.Parameter(512, 512)
        self.w3 = nn.Parameter(512, 5)
        self.loss_function = nn.SoftmaxLoss

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z0 = nn.Linear(xs[0], self.w1)
        h = z0
        for i in range(1, len(xs)):
            zi = nn.Add(nn.ReLU(nn.Linear(xs[i], self.w1)),
                        nn.ReLU(nn.Linear(h, self.w2)))
            h = zi
        output = nn.Linear(h, self.w3)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return self.loss_function(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size, start_lr, end_lr, delta_loss = 100, 8e-1, 1e-1, (1 / 350)
        gamma = (end_lr / start_lr) ** delta_loss
        iter, flagc, lr = unit_one, False, start_lr

        while not flagc:
            for x, y in dataset.iterate_once(batch_size):
                grads = nn.gradients(self.get_loss(x, y),
                        [self.w1, self.w2, self.w3])
                for w, grad in zip([self.w1, self.w2, self.w3], grads):
                    w.update(grad, -lr)
                if dataset.get_validation_accuracy() + 0.01 > 0.9:
                    flagc = True
                    break
                if iter < 3:
                    lr = lr * gamma
            iter = 1 + iter

class Attention(object):
    def __init__(self, layer_size, block_size):
        """
        Initializes the Attention layer.

        Arguments:
            layer_size: The dimensionality of the input and output vectors.
            block_size: The size of the block for the causal mask (used to apply causal attention).

        We initialize the weight matrices (K, Q, and V) using random normal distributions.
        The causal mask is a lower triangular matrix (a matrix of zeros above the diagonal, ones on and below the diagonal).
        """

        self.k_weight = np.random.randn(layer_size, layer_size)
        self.q_weight = np.random.randn(layer_size, layer_size)
        self.v_weight = np.random.randn(layer_size, layer_size)

        # Create the causal mask using numpy
        self.mask = np.tril(np.ones((block_size, block_size)))

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to the input tensor. This includes computing the query, key, and value matrices,
        calculating the attention scores, applying the causal mask, and then generating the output.

        Arguments:
            input: The input tensor of shape (batch_size, block_size, layer_size).

        Returns:
            output: The output tensor after applying the attention mechanism to the input.

        Remark: remember to use the causal mask and nn.softmax (in nn.py) will be helpful.
        """

        B, T, C = input.shape

        """YOUR CODE HERE"""
        Q = np.matmul(input, self.q_weight)  # Shape: (B, T, layer_size)
        K = np.matmul(input, self.k_weight)  # Shape: (B, T, layer_size)
        V = np.matmul(input, self.v_weight)  # Shape: (B, T, layer_size)

        # Compute scaled dot-product attention scores
        dk = self.layer_size
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(dk)  # Shape: (B, T, T)

        # Apply the causal mask
        mask = self.mask[:T, :T]
        masked_scores = np.where(mask == 1, scores, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = nn.softmax(masked_scores, dim=-1)  # Shape: (B, T, T)

        # Compute the output
        output = np.matmul(attention_weights, V)  # Shape: (B, T, layer_size)

        return output