from pylab import *

# R,Q = vgg_rq(S)  Just like qr but the other way around.
#
# If R,Q = vgg_rq(X), then R is upper-triangular, Q is orthogonal, and X==R*Q.
# Moreover, if S is a real matrix, then det(Q)>0.

# By awf

def vgg_rq(S):
    
    S = S.T
    Q, U = qr(fliplr(flipud(S)))
    Q = fliplr(flipud(Q.T))
    U = fliplr(flipud(U.T))
    
    if det(Q) < 0:
        t[:,0] = -t[:,0]
        
    return U, Q