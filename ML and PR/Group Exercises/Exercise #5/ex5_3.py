import numpy as np
import matplotlib.pyplot as plt

def log_loss(w, X, y):
    """ 
    Computes the log-loss function at w. The 
    computation uses the data in X with
    corresponding labels in y. 
    """
    
    L = 0 # Accumulate loss terms here.
        
    # Process each sample in X:
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(y[n] * np.dot(w, X[n])))
    
    return L
    
def grad(w, X, y):
    """ 
    Computes the gradient of the log-loss function
    at w. The computation uses the data in X with
    corresponding labels in y. 
    """
        
    G = 0 # Accumulate gradient here.
    
    # Process each sample in X:
    for n in range(X.shape[0]):
        
        numerator = np.dot(-y[n],np.dot(X[n],np.exp(np.dot(-y[n],np.dot(w.T,X[n])))))     # TODO: Correct these lines
        denominator = 1 + np.exp(np.dot(-y[n],np.dot(w.T,X[n])))   # TODO: Correct these lines
        
        G += numerator / denominator
    
    return G
    
if __name__ == "__main__":
        
    # Add your code here:
        
    # 1) Load X and y.        
    
    X = np.loadtxt('C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #5/log_loss_data/X.csv', delimiter=',', dtype= float, usecols =(0, 1))
    y = np.loadtxt('C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #5/log_loss_data/y.csv', delimiter=',', dtype= int, usecols =(0))

    # 2) Initialize w at w = np.array([1, -1])
    
    w = np.array([1, -1])
    
    # 3) Set step_size to a small positive value.
    
    step = 0.01

    # 4) Initialize empty lists for storing the path and    
    # accuracies: W = []; accuracies = []
    
    W = []
    accuracies = []
    
    for iteration in range(100):

        # 5) Apply the gradient descent rule.
        
        w = w - step * grad(w, X, y)

        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), log_loss(w, X, y)))
        
        # 7) Compute the accuracy (already done for you)
            
        # Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
                # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
                # Transform [0,1] coding to [-1,1] coding
        y_pred = 2*y_pred - 1

        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        
        W.append(w)
    
    # 8) Below is a template for plotting. Feel free to 
    # rewrite if you prefer different style.
    
    W = np.array(W)
    
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")