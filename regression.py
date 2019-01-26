import numpy as np

def softmax_function(vector):
    """
    Applies softmax transform
    """
    exp = [np.exp(i) for i in vector]
    sum_exp = np.sum(exp)
    return [j/sum_exp for j in exp]


#Used example dataset with 826 datapoints
#load dataset as numpy arrays
#with open('train.npy', 'rb') as fin:
#    X = np.load(data) #shape (826, 2)

#with open('test.npy', 'rb') as fin:
#    y = np.load(data) #shape (826,)

#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
#plt.show()
#visualization of dataset

def expand(X):
    """
    Adds quadratic features.
    This expansion allows your linear model to make non-linear separation.

    For each sample (row in matrix), computes an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]

    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))

    # TODO:<your code here>
    X_expanded[:, 0], X_expanded[:, 1] = X[:, 0], X[:, 1]
    X_expanded[:, 2], X_expanded[:, 3] = X[:, 0]**2, X[:, 1]**2
    X_expanded[:, 4], X_expanded[:, 5] = X[:, 0]*X[:, 1], np.ones(X.shape[0])

    return X_expanded
# X_expanded = expand(X)


### Logistic Regression ###
# To classify objects we will obtain probability of object belongs to class '1'.
# To predict probability we will use output of linear model and logistic function:
#𝑎(𝑥;𝑤)=⟨𝑤,𝑥⟩
#𝑃(𝑦=1∣∣𝑥,𝑤)=1/(1+exp(−⟨𝑤,𝑥⟩))=𝜎(⟨𝑤,𝑥⟩)
def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see function above

    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    a = np.dot(X, w)
    prob = 1 / (1 + np.exp(-a))
    prob = np.array(prob)
    return prob
## dummy_weights = np.linspace(-1, 1, 6)
## output= probability(X_expanded[:1, :], dummy_weights)[0]


# In logistic regression the optimal parameters $w$ are found by cross-entropy minimization:
# Loss for one sample: 𝑙(𝑥𝑖,𝑦𝑖,𝑤)=−[𝑦𝑖⋅𝑙𝑜𝑔𝑃(𝑦𝑖=1|𝑥𝑖,𝑤)+(1−𝑦𝑖)⋅𝑙𝑜𝑔(1−𝑃(𝑦𝑖=1|𝑥𝑖,𝑤))]
# Loss for many samples: 𝐿(𝑋,𝑦⃗ ,𝑤)=(1/ℓ)∑𝑙(𝑥𝑖,𝑦𝑖,𝑤)
def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function L using formula above.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    m = float(X.shape[0])
    a = probability(X, w)
    cross_entropy = y*np.log(a) + (1-y)*log(1-a)
    cost = -np.sum(cross_entropy)
    cost = cost/m
    loss = np.squeeze(cost) # remove single deminsional entries from shape
    assert(loss.shape == ()) # assure cost is tuple/scalar
    return loss
## dummy_weights = np.linspace(-1, 1, 6)
# output = compute_loss(X_expanded, y, dummy_weights)



#Since we train our model with gradient descent, I compute gradients.
#To be specific, we need a derivative of loss function over each weight [6 of them].
# ∇𝑤𝐿=1ℓ∑𝑖=1ℓ∇𝑤𝑙(𝑥𝑖,𝑦𝑖,𝑤)
#First figure out a derivative with pen and paper.
# Check math against finite differences to test:
#(estimate how 𝐿 changes if you shift 𝑤 by 10−5 or so).

def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], computes vector [6] of derivatives of L over each weights.
    Again keeping in mind that our loss is averaged over all samples (rows) in X.
    """
    m = float(X.shape[0])
    A = probability(X, w)
    dZ = A-y
    dw = np.dot(dZ, X)/m
    return dW
## dummy_weights = np.linspace(-1, 1, 6)
# output = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))

#function to visualize the predictions
#h = 0.01
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()
#visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])



### Training ###
# train our classifier we wrote using stochastic gradient descent

# Mini-batch SGD
#Stochastic gradient descent just takes a random batch of  𝑚  samples on each iteration, calculates a gradient of the loss on it and makes a step
# 𝑤𝑡=𝑤𝑡−1−𝜂(1/𝑚)∑∇𝑤𝑙(𝑥𝑖𝑗,𝑦𝑖𝑗,𝑤𝑡)
def mini_batch()
    np.random.seed(42)
    w = np.array([0, 0, 0, 0, 0, 1])
    eta = 0.1 # learning rate
    n_iter = 100
    batch_size = 4
    loss = np.zeros(n_iter)
    plt.figure(figsize = (12, 5))
    for i in range(n_iter):
        ind = np.random.choice(X_expanded.shape[0], batch_size)
        loss[i] = compute_loss(X_expanded, y, w)
        if i % == 0:
            visualize(X_expanded[ind, :], y[ind], w, loss)

        dW = compute_grad(X_expanded[ind, :], y[ind], w)
        w = w - eta*dW
    visualize(X, y, w, loss)
    plt.clf()
# output = compute_loss(X_expanded, y, w)

# SGD with momentum
# Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations.
# It does this by adding a fraction  𝛼  of the update vector of the past time step to the current update vector.
#𝜈𝑡=𝛼𝜈𝑡−1+𝜂(1/𝑚)∑∇𝑤𝑙(𝑥𝑖𝑗,𝑦𝑖𝑗,𝑤𝑡)
#𝑤𝑡=𝑤𝑡−1−𝜈𝑡
def sgd_momentum():
    np.random.seed(42)
    w = np.array([0, 0, 0, 0, 0, 1])
    eta = 0.05 # learning rate
    alpha = 0.9 # momentum
    nu = np.zeros_like(w)
    n_iter = 100
    batch_size = 4
    loss = np.zeros(n_iter)
    plt.figure(figsize=(12, 5))
    for i in range(n_iter):
        ind = np.random.choice(X_expanded.shape[0], batch_size)
        loss[i] = compute_loss(X_expanded, y, w)
        if i % 10 == 0:
            visualize(X_expanded[ind, :], y[ind], w, loss)

        dW = compute_grad(X_expanded[ind, :], y[ind], w)
        nu = alpha*nu + eta*dW
        w = w - nu
    visualize(X, y, w, loss)
    plt.clf()
#output = compute_loss(X_expanded, y, w)


# RMSprop
# Implemention of RMSPROP algorithm, which use squared gradients to adjust learning rate:
# 𝐺𝑡𝑗=𝛼𝐺^(𝑡−1)_𝑗 + (1−𝛼)𝑔^(2)_𝑡𝑗
# 𝑤^𝑡_𝑗=(𝑤^(𝑡−1)_𝑗) − (𝑔_𝑡𝑗)(𝜂)√(𝐺𝑡𝑗+𝜀)
def rms_prop():
    np.random.seed(42)
    w = np.array([0, 0, 0, 0, 0, 1.])
    eta = 0.1 # learning rate
    alpha = 0.9 # moving average of gradient norm squared
    g2 = np.zeros_like(w) # we start with None so that you can update this value correctly on the first iteration
    eps = 1e-8
    G = np.zeros_like(w)
    n_iter = 100
    batch_size = 4
    loss = np.zeros(n_iter)
    plt.figure(figsize=(12,5))
    for i in range(n_iter):
        ind = np.random.choice(X_expanded.shape[0], batch_size)
        loss[i] = compute_loss(X_expanded, y, w)
        if i % 10 == 0:
            visualize(X_expanded[ind, :], y[ind], w, loss)

        dW = compute_grad(X_expanded[ind, :], y[ind], w)
        G = (alpha*G) + (1-alpha)*(dW**2)
        w = w - (eta*dW)/(np.sqrt(G + eps))
    visualize(X, y, w, loss)
    plt.clf()

# output = compute_loss(X_expanded, y, w)
