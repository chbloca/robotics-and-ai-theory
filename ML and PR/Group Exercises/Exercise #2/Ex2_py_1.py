import numpy as np
import matplotlib.pyplot as plt

def synt_signal():
    f0 = 0.017
    w = np.sqrt(0.25)*np.random.randn(100)
    
    X = np.arange(100)
    y = np.sin(2*np.pi*f0*X) + w
    plt.plot(y)
    return y
    

def signal_estimation(y):
    scores = []
    frequencies = []
    for f in np.linspace(0, 0.5, 1000):
        # Create vector e. Assume data is in x.
        n = np.arange(100)
        z = -2*np.pi*1j*f*n
        e = np.exp(z)
        score = np.abs(np.dot(y, e))
        scores.append(score)
        frequencies.append(f)
    fHat = frequencies[np.argmax(scores)]
    print(fHat)    
    
if __name__ == "__main__":
    
    # Part 4
    y = synt_signal()
    signal_estimation(y)
