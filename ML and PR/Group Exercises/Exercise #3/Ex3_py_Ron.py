import numpy as np
import matplotlib.pyplot as plt

def SignalGen(f0):
    #3a Sin
    zeros_1 = np.zeros(500)
    zeros_2 = np.zeros(300)
    n = np.arange(100)
    x = np.concatenate((zeros_1, n, zeros_2), axis=0)
    y = np.sin(2 * np.pi * f0 * x)
    plt.figure(1)
    plt.subplot(411)
    plt.title("Raw Signal")
    plt.plot(y)
    #3b Noise+ Sin
    w = np.sqrt(0.5) * np.random.randn(y.size)
    y_n = y + w
    plt.subplot(412)
    plt.title("Signal + Noise")
    plt.plot(y_n)
    #3c Detectors
    #Method 1
    res = []
    wsz = 100
    zeros_3 = np.zeros(100)
    limit = len(y_n)-wsz
    for w in range(limit):
        wndw = y_n[w:w+wsz-1]
        y = np.convolve(np.cos(2*np.pi*f0*n), wndw, 'same')
        res = np.append(res, y)
    res = np.concatenate((res, zeros_3), axis=0)
    plt.subplot(413)
    plt.title("Method 1 - Known signal")
    plt.plot(res)
    #Method 2
    h = np.exp(-2 * np.pi * 1j * f0 * n)
    res2 = np.abs(np.convolve(h, y_n, 'same'))
    plt.subplot(414)
    plt.title("Method 2 - Random signal")
    plt.plot(res2)
    plt.show()
if __name__ == "__main__":
    SignalGen(0.1)
    #4 change frequency to 0.03
    SignalGen(0.03)

