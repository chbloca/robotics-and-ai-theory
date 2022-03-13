syms eX eY theta1 theta2

%fkine
eX=.75*cos(theta1)+.5*cos(theta1+theta2)
eY=.75*sin(theta1)+.5*sin(theta1+theta2)

e=[eX;eY]
%%
J=[diff(eX,theta1) diff(eX,theta2);
  diff(eY,theta1)  diff(eY,theta2)]
%%
goalE=[.75;.4];

%% Forward kinematics
currTheta=[pi/8;pi/8]; %Initial configuration
currE=subs(e,[theta1;theta2],currTheta)
currE=double(currE)
%%
error=100;
beta=0.1
while (error>.001)
    deltaE=beta*(goalE-currE)
    currJ=subs(J,[theta1;theta2],currTheta);
    currJ=double(currJ)
    invJ=inv(currJ)
    deltaTheta=invJ*deltaE; %
    currTheta=currTheta+deltaTheta
    currE=subs(e,[theta1;theta2],currTheta); % fkine
    currE=double(currE)
    error=norm(goalE-currE,2)
end

%% robotic toolbox
%check with robotic toolbox


