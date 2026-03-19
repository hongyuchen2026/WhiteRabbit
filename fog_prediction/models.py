from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    return model

class SigmoidActivation:
    """
    The sigmoid activation function

    You should not need to edit this class
    """
    def sigmoid(self, x):
        """
        Helper function to compute sigmoid and avoid warnings
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        First, save the input to self.input_ for gradient updates
        Then, return sigmoid(X)
        """
        self.input_ = X
        return self.sigmoid(X)

    def backward(self, grad, lr=None):
        """
        Using the saved inputs from the last `forward` call,
            compute the gradient of the sigmoid with respect to those inputs
            Multiply by `grad`, the gradient computed previously
        """
        new_grad = self.sigmoid(self.input_) * (1 - self.sigmoid(self.input_))
        return new_grad * grad


class ReluActivation:
    def forward(self, x):
        """
        First, save *a copy* of x to self.input_ for gradient updates
        Then, return ReLU(x) as a numpy calculation
        """

        self.input_ = x.copy()
        ReLU = np.where(x<0, 0, x)
        return ReLU


    def backward(self, grad, lr=None):
        """
        Using the saved inputs from the last `forward` call,
            compute the gradient of ReLU with respect to those inputs
            Multiply by `grad`, the gradient computed previously

        Notes:
          - The derivative of the ReLU is either 0
            if the previous input was <= 0, and otherwise 1.
          - The ReLU doesn't have any parameters to update,
            so you don't need to use `lr`
          - But do remember to include `grad` in your calculation!
        """

        new_grad = np.where(self.input_<0, 0, 1)
        return new_grad * grad


class FullyConnected:
    """
    A fully-connected layer 
    """
    def __init__(self, input_dim, output_dim, regularizer=None):
        """
        input_dim: the input dimension of the layer,
            *not* including the intercept that will be added
            If this is the first layer and the input is 2-dimensional,
            input_dim should be 2.
        output_dim: the output dimension of the layer
        regularizer: if not None, must implement `.grad(weights)`
            to be called in `self.backward()` to add regularization
            to this layer

        Note that self.weights will have shape [1 + input_dim, output_dim]
            because an intercept (or bias) term is added here and then
            an intercept column is to X whenever `self.forward(X)` is called
        """
        # A weight initialization strategy called "Xavier initialization"
        self.weights = np.random.normal(
            0, np.sqrt(1 / input_dim), [1 + input_dim, output_dim]) #[d+1, dnew]
        self.regularizer = regularizer

    def forward(self, X):
        """
        First, save the input to self.input_ for gradient updates
        Then compute X @ W
        """
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1) #[n, d+1]
        self.input_ = X
        return X.dot(self.weights) #[n, dnew]

    def backward(self, grad, lr=0.1):
        """
        Using saved inputs from the previous `forward` call, compute
            two gradients: d(X @ W)/dW and d(X @ W)/dX.
            The first is used to update self.weights, and the second
            is returned to be used by later `backward` calls in the network.

        Note: you should not need to edit this function.
        """

        # Make sure the gradient that's been computed so far
        #   matches up to the shape of this layer
        input_dim, output_dim = self.weights.shape #[d+1, dnew]
        batch_size, output_dim2 = grad.shape #[n, dnew]
        assert output_dim == output_dim2, "Shape mismatch"

        update = np.zeros_like(self.weights) #[d+1, dnew]
        new_grad = np.zeros([batch_size, input_dim - 1]) #[n, d]

        # Compute dL/dW for each node in this layer，
        for i in range(output_dim):
            update[:, i] = np.mean(grad[:, (i, )] * self.input_,    
                                   axis=0, keepdims=True) #[n, dnew] * [n, d+1] 变换后= [d+1, dnew]

        # Compute dL/dX for the next `backward` call
        #   and multiply it by `grad`
        new_grad = grad.dot(self.weights[1:, ].T) #[n, dnew] * [d, dnew].T = [n, d]

        # Now, we can update our weights
        self.weights -= lr * update #[d+1, dnew]

        # If using regularization, perform an additional update to self.weights
        if self.regularizer is not None:
            self.weights -= lr * self.regularizer.grad(self.weights)

        return new_grad

class NeuralNetwork:
    """
    A wrapper class for a neural network composed of
    layers and a loss function
    """
    def __init__(self, layers, loss, learning_rate=1):
        """
        layers: a list of layers
            each must have a `forward` and `backward` function
        loss: the loss function to use when calling self.backward

        You should not need to edit this function.
        """
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def predict(self, X):
        """
        Helper function to match the scikit-learn API

        You will not need to edit this function.
        """
        return self.forward(X)

    def forward(self, X):
        """
        Take the input and pass it forward through each layer of the network,
        using the `.forward()` function of each layer.

        Return the output of the final layer.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.out = out
        return out

    def backward(self, pred, y):
        """
        Take the predicted and target outputs and compute the loss.

        Then, beginning with `self.loss` and continuing *backwards*
        through each layer of the network, use the `.backward()`
        function of each layer to perform backpropagation.

        Note: each call to `backward()` in self.layers
            should use self.learning_rate

        Returns None
        """
        self.loss.forward(pred, y)
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)
        return None
        
    def fit(self, X, y, max_iter=10000):
        """
        Train the model on the data for `max_iter` iterations.
        For each iteration, call `self.forward` and then `self.backward` to
            make a prediction and then update each layer's weights.  Each
            layer needs to have its `forward()` function called before its
            `backward()` function; this should be handled by this class's
            `self.forward()` and `self.backward()`.

        This function should always run for `max_iter` iterations; don't stop
            early even if the gradients are negligibly small.

        Returns None
        """
        for i in range(max_iter):
            pred = self.forward(X)
            self.backward(pred,y)
        return None

class BinaryCrossEntropyLoss:
    def forward(self, y_pred, y_true):
        """
        Save the inputs to self.input_ and then
            compute the binary cross-entropy loss

        You will not need to edit this function.
        """
        assert set(np.unique(y_true)).issubset(set([0, 1]))
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1) #[n,1]
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1) #[n,1]

        self.input_ = (y_pred, y_true)
        grad = np.where(y_true, -np.log(y_pred), -np.log(1 - y_pred))
        return np.mean(grad)

    def backward(self, grad=None, lr=None):
        """
        Compute the gradient of the loss function
        `grad` and `lr` are left as arguments to match the other
            `backward` functions, but will never be passed anything.

        You will not need to edit this function.
        """
        assert grad is None
        (y_pred, y_true) = self.input_
        grad = (- y_true + y_pred) / (y_pred - y_pred ** 2)
        return grad


class SquaredLoss:
    def forward(self, y_pred, y_true):
        """
        Save the inputs to self.input_ and then compute the **mean** squared
        error loss. The output of this should be a single scalar.

        While sometimes MSE is written including a factor of 1/2, because
          `d/dx (1/2 x ^ 2) = x`, please do not include that factor here.
        """
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)

        self.input_ = (y_pred, y_true)
        squared_loss = np.mean((y_pred - y_true)**2).item()
        return squared_loss

    def backward(self, grad=None, lr=None):
        """
        Compute the gradient of the squared loss. Please **do not** use np.mean
        in your calculations here! We want to track the gradient of the loss
        with respect to every example; we will aggregate the loss across
        examples before we update parameters.

        You should use the arguments saved to self.input_ from the last time
            `forward()` was called.

        `grad` and `lr` are left as arguments to match the other `backward`
            functions. However, `grad` should never be passed to this function,
            as the loss function is the first calculation in the
            backpropagation sequence. You should not use `lr` in this function,
            as no parameters are being updated yet.
        """
        assert grad is None

        y_pred, y_true = self.input_
        grad_squared_loss = (y_pred - y_true)*2
        return grad_squared_loss #[d, 1]

class Regularizer:
    """
    Regularization for FullyConnected layers
    """
    def __init__(self, alpha=0.01, penalty='l2'):
        """
        penalty: type of distance measure, either "l1" or "l2"
        alpha: weight parameter for the regularization gradient

        You will not need to edit this function.
        """
        self.alpha = alpha
        self.penalty = penalty

    def grad(self, weights):
        weights = weights.copy()  # avoid direct modification
        if self.penalty == "l1":
            return self.l1_grad(weights)
        elif self.penalty == "l2":
            return self.l2_grad(weights)

    def l1_grad(self, weights):
        """
        Compute the *gradient* of L1 regularization
            with respect to the given weights.

        L1 regularization is just the absolute value, so the partial gradient
            for a given weight is either 1, -1, or 0 depending on whether the
            weight is greater than, less than, or equal to 0.

        Note: weights[0, :] contains the intercept, and you should not apply
            regularization to those parameters.
        """
        
        grad = np.where(
            weights > 0,  1,
            np.where(weights < 0, -1, 0)
        )

        grad[0, :] = 0
        return grad*self.alpha

    def l2_grad(self, weights):
        """
        Compute the *gradient* of L2 regularization
            with respect to the given weights.

        L2 regularization is the squared value, so the partial gradient
            for a given weight should be proportional to that weight

        Note: weights[0, :] contains the intercept, and you should not apply
            regularization to those parameters.
        """
        grad = weights*2
        grad[0, :] = 0
        return grad*self.alpha
    
class LogisticRegression():
    def __init__(self, learning_rate=1e-1, max_iter=200):
        """
        A logistic regression classifier. This binary classifier learns a
        linear boundary that separates input space into two, such that points
        on one side of the line are one class and points on the other side are
        the other class.

        Read the `logistic_regression.pdf` handout in the base folder of your
        repository before trying to implement this! 

        Args:
            max_iter (int): the algorithm stops after this many iterations if
                it has not converged.

            learning_rate (float): how large of a gradient step to take at each
                update.

        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the model to the data. You should not have to modify this
        function -- all your work should go in `update_weights` and `predict`.

        Note: self.add_intercept is called to add an intercept to the features

        Args:
            X (np.ndarray): a NxK array containing N examples each with K features.
            y (np.ndarray): a Nx1 array containing binary targets.
        Returns:
            n_iters: the number of iterations the model took to converge,
                or self.max_iter
        """
        X2 = self.add_intercept(X) #[n, d+1]
        self.weights = np.zeros((X2.shape[1], 1)) #[d+1, 1]

        for n_iters in range(1, 1 + self.max_iter):
            stop = self.update_weights(X2, y)
            if stop:
                break

        return n_iters

    def add_intercept(self, X):
        """
        Helper function to add a column of 1's to your features
        """
        return np.concatenate([np.ones([X.shape[0], 1]), X], axis=1) #[n, d+1]

    def update_weights(self, X, y):
        """
        Perform one iteration of gradient descent for LogisticRegression
        Note: don't forget to use `self.learning_rate` to scale the amount of
            the update.

        Pseudocode:
            for each example in X
                compute the gradient of binary cross entropy loss update the weights to better classify the data
            return whether the model has converged

        Args:
            X: the Nx(K+1) matrix of features, including an intercept
            y: the Nx1 array of targets

        Returns:
            stop: Boolean indicating whether the model has converged
            (Do not return the weights; update those in-place)
        """
        sigmoid = SigmoidActivation()
        w = self.weights
        h = sigmoid.sigmoid(X @ w)
        grad = X.T @ (h - y) / X.shape[0]
        w = w - self.learning_rate * grad
        self.weights = w
        if np.all(np.absolute(grad)<=0.01) :
            return True
        else:
            return False

    def predict(self, X):
        """
        Given features, a 2D numpy array, use the trained model to predict
        target classes. Call this after calling fit.

        Note: Keep the `self.add_intercept` code to ensure you include the
            intercept

        Args:
            X (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of trained model on features,
                with predictions as {0, 1} labels.
        """
        sigmoid = SigmoidActivation()
        X = self.add_intercept(X)
        w = self.weights
        y0 = sigmoid.sigmoid(X @ w)
        predictions = np.where(y0>=0.5, 1, 0)
        return predictions