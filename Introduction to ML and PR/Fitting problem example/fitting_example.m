t=0:0.1:10;
y=sin(2*pi*3/10*t);
plot(t,y)
nnstart
%you train the first NN with 2 neurons, and then another one with 8 neurons
plot(t,y,'r',t,net(t),'g',t,net1(t),'b')