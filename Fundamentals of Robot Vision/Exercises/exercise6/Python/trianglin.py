#from pylab import *
import numpy as np

def trianglin(P1, P2, x1, x2):
    # Inputs:
    #   P1 and P2, projection matrices for both images
    #   x1 and x2, image coordinates for both images
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    #X = np.ones(4)  # replace with your implementation
    aux1 = np. zeros((3,3))
    aux1[0,1] = -x1[2]
    aux1[0,2] = x1[1]
    aux1[1,0] = x1[2]
    aux1[1,2] = -x1[0]
    aux1[2,0] = -x1[1]
    aux1[2,1] = x1[0]
    
    aux2 = np. zeros((3,3))
    aux2[0,1] = -x2[2]
    aux2[0,2] = x2[1]
    aux2[1,0] = x2[2]
    aux2[1,2] = -x2[0]
    aux2[2,0] = -x2[1]
    aux2[2,1] = x2[0]
    
    print(aux1.shape)
    print(aux2.shape)
    print(P1.shape)
    print(P2.shape)
    
    temp1 = np.dot(aux1,P1)
    temp2 = np.dot(aux2,P2)
    
    A = np.vstack((temp1,temp2))    
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(A.T,A))
    X = eigen_vectors[np.argmin(eigen_values)]
    ##-your-code-ends-here-##
    
    return X