from torch.nn.functional import mse_loss, cross_entropy
import torch.nn.functional as F

def regression_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    Inputs:
        y_pred: a node with shape (batch_size x 1), containing the predicted y-values
        y: a node with shape (batch_size x 1), containing the true y-values
            to be used for training
    Returns: a tensor of size 1 containing the loss
    """
    "*** YOUR CODE HERE ***"
    return ((y_pred - y) ** 2).mean()



def digitclassifier_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred: a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """


def languageid_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a node with shape
    (batch_size x 5). Each row is a one-hot vector encoding the correct
    language.

    Inputs:
        model: Pytorch model to use
        y_pred: a node with shape (batch_size x 5)
        y: a node with shape (batch_size x 5)
    Returns: a loss node
    """
    "*** YOUR CODE HERE ***"
    y_true = y.argmax(dim=1)
    loss = F.cross_entropy(y_pred, y_true)
    return loss


def digitconvolution_Loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred : a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    
