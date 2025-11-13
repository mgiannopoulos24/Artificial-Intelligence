from torch.nn.functional import mse_loss, cross_entropy, log_softmax
from torch import sum, mean

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
    # Apply log_softmax to the predictions (logits)
    log_probs = log_softmax(y_pred, dim=1)
    # Calculate loss: - sum(true_label * log_probability)
    loss_per_sample = -sum(y * log_probs, dim=1)
    return mean(loss_per_sample)


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
    # Convert one-hot to class indices for cross_entropy
    target = y.argmax(dim=1)
    return cross_entropy(y_pred, target)


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
    return cross_entropy(y_pred, y)
    
