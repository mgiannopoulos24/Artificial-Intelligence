from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
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
        training_complete = False
        while not training_complete:
            training_complete = True
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                pred = model.get_prediction(x)
                if pred.item() != y.item():
                    # Update weights: w = w + (y * x)
                    model.w += y * x
                    training_complete = False


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.05)  # Use Adam instead of SGD
    num_epochs = 5000
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            # Forward pass
            pred = model(x)
            loss = regression_loss(pred, y)
            epoch_loss += loss.item()
            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < 0.02:
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """


def train_languageid(model, dataset):
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
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        epoch_loss = 0.0
        for batch in dataloader:
            x, y = batch['x'], batch['label']
            optimizer.zero_grad()
            y_pred = model(x)
            loss = digitconvolution_Loss(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")
