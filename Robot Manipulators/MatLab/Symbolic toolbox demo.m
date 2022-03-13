
%%
% Here you can see the use of: 
%symbolic toolbox: subs, simplify
%Transformation matrices: transl,trotz,trotx(),troty()
%
%
%transl([x y z])
%trotz()
%trotx()
%troty()
%simplify (transl([x y z]))

%%
%Computing the Homogenous transformation matrix from frame i, to i-1 using
%the DH parameters
%For planar manipulator
syms alpha theta d a
Ti2iminus=trotx(alpha)*transl([a 0 0])*transl([0 0 d])*trotz(theta)
Ti2iminus=simplify (Ti2iminus)

%%
%From frame 2 to frame 1 when theta2=0
T21=subs(Ti2iminus,[alpha a d theta],[0 .75 0 0])
T21=double(T21)
%From frame 1 to frame 0 when theta1=90
T10=subs(Ti2iminus,[alpha a d theta],[0 0 0 pi/2])

%From Frame 2 to frame 0 when theta1=90 and theta2=0

T20=T10*T21

%%
%Forward kinematics as function of theta1 and theta2
syms theta1 theta2
T21=subs(Ti2iminus,[alpha a d theta],[0 .75 0 theta2])
%From frame 1 to frame 0 when theta1=90
T10=subs(Ti2iminus,[alpha a d theta],[0 0 0 theta1])

T20=simplify(T10*T21)

%%
