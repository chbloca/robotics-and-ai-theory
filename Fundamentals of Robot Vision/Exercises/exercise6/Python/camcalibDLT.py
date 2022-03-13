import numpy as np

# Direct linear transformation (DLT) is an algorithm which 
# solves a set of variables from a set of similarity relations,
# e.g  the relation between 3D points in a scene and 
# their projection onto an image plane

def camcalibDLT(Xworld, Xim):
    # Inputs: 
    #   Xworld, world coordinates in the form (id, coordinates)
    #   Xim, image coordinates in the form (id, coordinates)
    
    # Create the matrix A 
    ##-your-code-starts-here-##
    A = []

    #Ay = []
    #Ax = []


    for i in range(0, 8):
        Ay_i = np.hstack((np.zeros(4), Xworld[i,:], -Xim[i,1]*Xworld[i,:]))
        Ax_i = np.hstack((Xworld[i,:], np.zeros(4), -Xim[i,0]*Xworld[i,:]))
        A.append(Ay_i)
        A.append(Ax_i)
    A = np.array(A)
    print(A.shape)
    
    #A = np.vstack((Ay,Ax)).reshape((-1,),order='F')

    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting,
    # the best solution is given by the eigenvector of 
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(A.T,A))
    ev = eigen_vectors[:,np.argmin(eigen_values)]
    print(ev.shape)
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3,4))  # uncomment this
    #P = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float) # remove this
    
    return P