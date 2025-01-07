from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module, Linear, MSELoss, Tanh
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import numpy as np


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        self.w = Parameter(torch.ones(1,dimensions)) #Initialize your weights here

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

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return torch.matmul(self.w, x.T).squeeze()


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        return 1 if score >= 0 else -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            converged = False
            while not converged:
                converged = True
                for batch in dataloader:
                    x, label = batch['x'], batch['label']
                    prediction = self.get_prediction(x)
                    if prediction != label:
                        converged = False
                        direction = label * x  # Update rule: w += direction * magnitude
                        self.w += direction



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        self.hidden_layer1 = Linear(1, 128)  # First hidden layer with 128 units
        self.hidden_layer2 = Linear(128, 64)  # Second hidden layer with 64 units
        self.output_layer = Linear(64, 1)  # Output layer
        self.activation = Tanh()  # Use Tanh for better approximation of periodic functions

        # Initialize weights
        xavier_uniform_(self.hidden_layer1.weight)
        xavier_uniform_(self.hidden_layer2.weight)
        xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = self.activation(self.hidden_layer1(x))  # First hidden layer
        x = self.activation(self.hidden_layer2(x))  # Second hidden layer
        return self.output_layer(x)  # Output layer

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predictions = self.forward(x)  # Get model predictions
        loss_fn = MSELoss()  # Mean squared error loss
        return loss_fn(predictions, y)  # Compute and return the loss
 
  

    def train(self, dataset, batch_size=32, learning_rate=0.01, max_epochs=1000, target_loss=0.02):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=learning_rate)  # Use Adam optimizer

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for sample in dataloader:
                x, y = sample['x'], sample['label']
                optimizer.zero_grad()  # Clear gradients
                loss = self.get_loss(x, y)  # Compute loss
                loss.backward()  # Backpropagate
                optimizer.step()  # Update parameters
                epoch_loss += loss.item()
            
            # Calculate average loss for the epoch
            epoch_loss /= len(dataloader)
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")

            # Stop training if target loss is achieved
            if epoch_loss <= target_loss:
                print("Target loss achieved. Stopping training.")
                break

class DigitClassificationModel(Module):
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
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        hidden_size = 128      # Hidden layer size
        # Define the layers of the network
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # Input to hidden
        self.fc2 = torch.nn.Linear(hidden_size, output_size) # Hidden to output



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        logits = self.fc2(x)         # Output layer (no activation function here)
        return logits

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        logits = self.run(x)  # Get the predicted logits from the model
        criterion = torch.nn.CrossEntropyLoss()  # Use cross-entropy loss
        loss = criterion(logits, y)  # Compute the loss between logits and true labels
        return loss

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        # Hyperparameters
        epochs = 10
        learning_rate = 0.001
        batch_size = 64

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer (e.g., Adam optimizer)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, y = batch['x'], batch['label']

                optimizer.zero_grad()  # Clear gradients from the last step
                loss = self.get_loss(x, y)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the model parameters

                total_loss += loss.item()  # Track the total loss for the epoch

            # Compute validation accuracy after each epoch if needed
            validation_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}, Validation Accuracy: {validation_accuracy}")

            # Stop early if validation accuracy is high enough
            if validation_accuracy > 0.98:
                break



class LanguageIDModel(Module):
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
        super(LanguageIDModel, self).__init__()

        self.num_languages = len(self.languages)
        "*** YOUR CODE HERE ***"
        self.hidden_size = 128

        # Layers
        self.embedding = nn.Linear(self.num_chars, self.hidden_size)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=False)
        self.output_layer = nn.Linear(self.hidden_size, self.num_languages)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
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
        batch_size = xs[0].shape[0]
        h = torch.zeros(1, batch_size, self.hidden_size, device=xs[0].device)  # Initialize hidden state

        for x in xs:
            x_embedded = self.embedding(x).unsqueeze(0)  # Add sequence dimension
            out, h = self.gru(x_embedded, h)

        logits = self.output_layer(h.squeeze(0))  # (batch_size x num_languages)
        return logits
        

    
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
        logits = self.run(xs)  # Forward pass
        loss = F.cross_entropy(logits, y.argmax(dim=1))  # Cross-entropy loss
        return loss


    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = Adam(self.parameters(), lr=0.001)

        for epoch in range(20):
            epoch_loss = 0.0
            for batch in dataloader:
                x = movedim(batch['x'], 1, 0)  # Transpose batch to (seq_len, batch_size, num_chars)
                y = batch['label']
                
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Calculate average loss and validation accuracy for this epoch
            epoch_loss /= len(dataloader)
            validation_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Validation Accuracy = {validation_accuracy:.2%}")
            
            # Early stopping if validation accuracy is high enough
            if validation_accuracy >= 0.85:
                print("Target accuracy achieved. Stopping training.")
                break

        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    "*** YOUR CODE HERE ***"
    input_height, input_width = input.shape
    weight_height, weight_width = weight.shape
    output_height = input_height - weight_height + 1
    output_width = input_width - weight_width + 1
    output = torch.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = input[i:i+weight_height, j:j+weight_width]
            output[i, j] = (region * weight).sum()

    return output
    
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        self.hidden_layer = Linear(26 * 26, 128)  # 26x26 is the output size after applying 3x3 convolution on 28x28 input
        self.output_layer = Linear(128, output_size)


    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        x = relu(self.hidden_layer(x))
        return self.output_layer(x)

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predictions = self.forward(x)
        return cross_entropy(predictions, y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = Adam(self.parameters(), lr=0.001)

        for epoch in range(20):
            epoch_loss = 0.0
            for sample in dataloader:
                x, y = sample['x'], sample['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")
 