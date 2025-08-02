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
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score_node = self.run(x)
        score = nn.as_scalar(score_node)
        if score >= 0:
            return 1
        else:
            return -1
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            mistakes = 0
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                true_label = nn.as_scalar(y)

                if prediction != true_label:
                    mistakes += 1
                    # Update the weights using the perceptron update rule
                    # w <- w + x * true_label
                    self.w.update(x, true_label)
            
            # If a full pass over the data is completed with no mistakes, training is done.
            if mistakes == 0:
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
         # Hyperparameters based on the provided suggestions
        self.learning_rate = 0.05
        self.batch_size = 200
        hidden_layer_size = 512

        # Layer 1: from 1 input feature to `hidden_layer_size`
        self.w1 = nn.Parameter(1, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        
        # Layer 2: from `hidden_layer_size` to 1 output feature
        self.w2 = nn.Parameter(hidden_layer_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # First layer transformation
        xw1 = nn.Linear(x, self.w1)

        biased_xw1 = nn.AddBias(xw1, self.b1)
        relu_output = nn.ReLU(biased_xw1)
        
        # Second layer transformation (output layer)
        xw2 = nn.Linear(relu_output, self.w2)
        predicted_y = nn.AddBias(xw2, self.b2)
        
        return predicted_y

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
        # Get the model's predictions for x
        predicted_y = self.run(x)
        # Compute the square loss between predictions and true values
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Collect all trainable parameters
        params = [self.w1, self.b1, self.w2, self.b2]
        
        # Loop until the desired loss is achieved
        while True:
            # Iterate through the dataset in batches
            for x, y in dataset.iterate_once(self.batch_size):
                # Compute the loss for the current batch
                loss = self.get_loss(x, y)
                
                # Compute gradients of the loss with respect to the parameters
                gradients = nn.gradients(loss, params)
                
                # Update each parameter using its gradient
                # The learning rate is negated for gradient descent
                for i in range(len(params)):
                    params[i].update(gradients[i], -self.learning_rate)


            # We iterate again to compute the total loss without updating weights
            total_loss = 0
            total_samples = 0
            for x, y in dataset.iterate_once(self.batch_size):
                # Get the batch loss and convert it to a Python number
                batch_loss_node = self.get_loss(x, y)
                batch_loss = nn.as_scalar(batch_loss_node)
                
                # Weight the batch loss by the number of samples in it
                num_samples_in_batch = x.data.shape[0]
                total_loss += batch_loss * num_samples_in_batch
                total_samples += num_samples_in_batch
            
            average_loss = total_loss / total_samples
            
            # If the average loss is at or below the target, stop training
            if average_loss <= 0.02:
                break

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
        # Hyperparameters based on the provided suggestions
        self.learning_rate = 0.5
        self.batch_size = 100
        hidden_layer_size = 200
        
        # Layer 1: from 784 input features (28x28 pixels) to `hidden_layer_size`
        self.w1 = nn.Parameter(784, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)

        # Layer 2: from `hidden_layer_size` to 10 output classes
        self.w2 = nn.Parameter(hidden_layer_size, 10)
        self.b2 = nn.Parameter(1, 10)

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
        # First layer transformation and activation
        xw1 = nn.Linear(x, self.w1)
        biased_xw1 = nn.AddBias(xw1, self.b1)
        relu_output = nn.ReLU(biased_xw1)

        # Second layer (output) transformation. No ReLU here as per instructions.
        xw2 = nn.Linear(relu_output, self.w2)
        logits = nn.AddBias(xw2, self.b2)

        return logits

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
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        params = [self.w1, self.b1, self.w2, self.b2]

        # Loop until the validation accuracy reaches the desired threshold
        while True:
        
            for x, y in dataset.iterate_once(self.batch_size):
      
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, params)

                for i in range(len(params)):
                    params[i].update(gradients[i], -self.learning_rate)

            validation_accuracy = dataset.get_validation_accuracy()

            if validation_accuracy >= 0.975:
                break

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

        "*** YOUR CODE HERE ***"
        # Hyperparameters
        self.hidden_size = 400
        self.learning_rate = 0.08
        self.batch_size = 100
        output_hidden_layer_size = 128

        # RNN parameters for the recurrent step
        self.W_x = nn.Parameter(self.num_chars, self.hidden_size)
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_hidden = nn.Parameter(1, self.hidden_size)

        # Parameters for the first output hidden layer
        self.W_output1 = nn.Parameter(self.hidden_size, output_hidden_layer_size)
        self.b_output1 = nn.Parameter(1, output_hidden_layer_size)
        
        # Parameters for the final output layer
        self.W_output2 = nn.Parameter(output_hidden_layer_size, len(self.languages))
        self.b_output2 = nn.Parameter(1, len(self.languages))

        # A list of all parameters for easy access during training
        self.params = [self.W_x, self.W_hidden, self.b_hidden, 
                       self.W_output1, self.b_output1, 
                       self.W_output2, self.b_output2]

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
        h = None

        # RNN loop
        for x in xs:
            linear_x = nn.Linear(x, self.W_x)
            if h is None:
                z = linear_x
            else:
                linear_h = nn.Linear(h, self.W_hidden)
                z = nn.Add(linear_x, linear_h)
            
            z_biased = nn.AddBias(z, self.b_hidden)
            h = nn.ReLU(z_biased)

        # Deeper output network
        # First output hidden layer
        output_hidden = nn.Linear(h, self.W_output1)
        output_hidden_biased = nn.AddBias(output_hidden, self.b_output1)
        output_hidden_activated = nn.ReLU(output_hidden_biased)

        # Final output layer (scores/logits)
        final_logits = nn.Linear(output_hidden_activated, self.W_output2)
        scores = nn.AddBias(final_logits, self.b_output2)

        return scores

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
        scores = self.run(xs)
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Set a slightly higher threshold to be safe
        stopping_threshold = 0.90

        while True:
            for xs, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, self.params)

                for i in range(len(self.params)):
                    self.params[i].update(gradients[i], -self.learning_rate)

            validation_accuracy = dataset.get_validation_accuracy()
            
            if validation_accuracy >= stopping_threshold:
                break
