%%

%Dynamics solved with Newton-Euler formulation

clear all

clc

syms t1 st1 at1 L1 g m1

syms t2 st2 at2 L2 m2

%Notation

% t1: theta 1

% st1: angular speed theta 1

% at1: angular acceleration theta 1

%Outward computations

% w1: angular speed for frame 1

% dw1: angular acceleration for frame 1

% dV1: linear acceleration for frame 1

% dVc1: linear acceleration for center of mass 1

% F1: force in center of mass 1

% N1: torque in center of mass 1

%Inward computations

% f1: force in joint 1

% n1: torque in joint 1

%OUTWARD ITERATIONS

%For link 1

w1=[0;0;st1];

dw1=[0;0;at1];

Pc1=[L1;0;0];

dV1=[g*sin(t1);g*cos(t1);0];

dVc1=cross(dw1,Pc1)+cross(w1,cross(w1,Pc1))+dV1;

% Forces and torques computed at center of mass_1

F1=m1*dVc1;

N1=[0;0;0];

%For link 2

R12=[cos(t2) sin(t2) 0; -sin(t2) cos(t2) 0; 0 0 1];

w2=R12*w1+[0;0;st2]

dw2=R12*dw1+cross(R12*w1,[0;0;st2])+[0;0;at2]

P21=[L1;0;0];

dV2=R12*(cross(dw1,P21)+cross(w1,cross(w1,P21))+dV1)

dV2=simplify(dV2)

Pc2=[L2;0;0];

dVc2=cross(dw2,Pc2)+cross(w2,cross(w2,Pc2))+dV2

dVc2=simplify(dVc2)

% Forces and torques computed at center of mass_2

F2=m2*dVc2

N2=[0;0;0]

%%

%%INWARD ITERATIONS

% Forces and torques computed for joint 2

f2=F2;

n2=cross(Pc2,F2)

n2=simplify(n2)

R21=R12.'

% Forces and torques computed for joint 1

f1=R21*f2+F1

f1=simplify(f1)

n1=N1+R21*n2+cross(Pc1,F1)+cross(P21,R21*f2)

n1=simplify(n1)