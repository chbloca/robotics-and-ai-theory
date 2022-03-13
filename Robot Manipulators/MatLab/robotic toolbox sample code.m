%%This code shows basic modelling and functions
%Of the Robotic Toolbox by Peter Corke
%%
%Defining link objects with the DH parameters. 
%check order of parameters
L(1)= Link([0 0 0 0],'modified')
L(2)= Link([0 0 .75 0],'modified')

%Defining serial manipulator
two_link=SerialLink(L,'name','two_link')
two_link

%%
clc
L(1)= Link([0 0 0 0 1],'modified')
two_link=SerialLink(L,'name','two_link')
%%
%Visualising: plot and teaching
two_link.teach([0,0])
%two_link.plot([0,0])
%%
%Acces tool property
two_link.tool
%Create tool HT matrix
toolHT=transl(.5,0,0)
%Attach new tool to the robot
two_link.tool=toolHT
%See effects of new tool
two_link.teach([0,0])

%%
%Show with second joint rotated 90 degrees
two_link.teach([0,pi/2])
%Can you compute mentally the new position of the manipulator?
%Verifying it with forward kinematics
two_link.fkine([0,pi/2])
%%
%Inverse kinematics
%Creating a target
target=transl(.75,.4,0)
%For rotating use trotx,troty,trotz

%Solve inverse kinematics
Q=two_link.ikine(target,[pi/2,pi/2],[1,1,0,0,0,0])

%Plot manipulator using angles obtained with the inverse kinematics
two_link.plot(Q)
%Verify result
two_link.fkine(Q)

%%

%Solve inverse kinematics
%ikunc, ikine6s
%ikine, numerical solution

Q=two_link.ikine(target,[pi/2,pi/2],'mask',[1,1,0,0,0,0])
Qd=Q*180/pi

%%
%L(3).qlim=[0,2] specifies prismatic joint stroke range
