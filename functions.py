def compute_loss(y, tx, w):
    """Calculate the loss (MSE)."""
    N= tx.shape[0]
    e = y - np.matmul(tx, w)
    return np.sum(e * e)/ (2 * N)

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def grad_logistic(y, x, w):
    z = sigmoid(x.dot(w))
    y=y.reshape(z.shape)
    return x.T.dot(z-y)/x.shape[0]


def compute_logi_loss(y, tx, w):
    y_t = sigmoid(tx.dot(w))
    N= len(y)
    loss1 = y.T.dot(np.log(y_t))
    loss2 = (1-y).T.dot(np.log(1-y_t))
    return -1/N*(loss1+loss2)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N= tx.shape[0]
    e = y - np.matmul(tx, w)
    return  -1 * np.dot(tx.T, e) /N 

def leasts_squares(y, tx):
    X2 = np.matmul(np.transpose(tx), tx)
    X2_inv = np.linalg.inv(X2)
    W = np.matmul(X2_inv, np.transpose(tx))
    W = W.dot(y)
    loss = compute_loss(y, tx, W)
    return W, loss

def ridge_regression(y, tx, lambda_):
    X2 = np.matmul(np.transpose(tx), tx) + lambda_ * np.identity(tx.shape[1])
    X2_inv = np.linalg.inv(X2)
    W = np.matmul(X2_inv, np.transpose(tx))
    # y_p = W.dot(y)
    loss = compute_loss(y, tx, W)
    return W, loss

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_gradient(y, tx, w)
        w = w- gamma * dw

        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss(y, tx, w)
    return w, loss

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size = batch_size, shuffle=True):
            dw = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w -gamma * dw

        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss(y, tx, w)    
    return w, loss