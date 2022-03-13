clc
clear all 

%Assigment 1 Robot Manipulating
disp('Lorenzo Etchenique 282974')
disp('Jan Dierkes 282910 ')
disp('Christen Blom-Dahl  282803')

%% R1:Transformations between links

disp('R1:Transformations between links')
syms alpha theta d a;
Ti2iminus=trotx(alpha)*transl([a 0 0])*transl([0 0 d])*trotz(theta);
Ti2iminus=simplify(Ti2iminus);
%parameters of the links
L1=.4;
L2=L1;
%parameters of the joins 
syms theta1 theta2 d3 theta4;
%From frame T(tool) to frame 4 
TT4=subs(Ti2iminus,[alpha a d theta],[0 .03 -.135 0]);
%From frame 4 to frame 3 when theta4=0
T43=subs(Ti2iminus,[alpha a d theta],[0 0 0 theta4]);
%From frame 3 to frame 2 when d3=0
T32=subs(Ti2iminus,[alpha a d theta],[0 L2 d3 0]);
%From frame 2 to frame 1 when theta2=0
T21=subs(Ti2iminus,[alpha a d theta],[0 L1 0 theta2]);
%From frame 1 to frame 0 when theta1=0
T10=subs(Ti2iminus,[alpha a d theta],[0 0 0 theta1]);
%From frame T to 0
TT0=simplify(T10*T21*T32*T43*TT4)


%% R2: HT with joints values =0

disp('R2: HT with joints values =0')
HTinitial0=subs(TT0,[theta1 theta2 d3 theta4],[0 0 0 0]);
%TB1 add the desplacement of the base of the SCARA robot
TB1=[1 0 0 .1;0 1 0 .1;0 0 1 .36; 0 0 0 1];
HTinitial=TB1*HTinitial0;
HTinitial=double(HTinitial)
%Position of the manipulator (x, y, z) with joints values=0
Pm=HTinitial*[0;0;0;1];
Pm=Pm(1:3);
Pm=double(Pm)


%% R3:Robot model in robotic toolbox

disp('R3:Robot model in robotic toolbox')
%Defining link objects with the DH parameters.
L(1)= Link([0 0 0 0],'modified');
L(2)= Link([0 0 L1 0],'modified');
L(3)= Link('theta', 0, 'a', L2, 'alpha', 0,'modified');
L(4)= Link([0 0 0 0],'modified');
L(3).qlim=[-0.225,0];
%Defining serial SCARA manipulator
SCARA=SerialLink(L,'name','links')
configuration=SCARA.config

%% R4: Robot plot in zero position

disp('R4: Robot plot in zero position')
W = [0 1.1 0 1.1 0 0.5];
W1 = [-.1 1 -.1 1 -.1 .1];
Theta1=0
Theta2=0 
D3=0 
Theta4=0
figure(1)
SCARA.plot([Theta1 Theta2 D3 Theta4],'workspace',W1,'scale', 2)


%% R5: Tool transformation

disp('R5: Tool transformation')
SCARA.tool;
%Create gripper
gripper=transl([.03 0 -0.135])
%Attach gripper to SCARA robot
SCARA.tool=gripper;


%% R6: Base transformation

disp('R6: Base transformation')
base=transl([.1 .1 .36])
SCARA.base=base;


%% R7: Robot in zero position

disp('R7: Robot in zero position' )
Theta1=0
Theta2=0 
D3=0 
Theta4=0
%End frame homogenous transformation matrix in zero position
HTzero_position = SCARA.fkine([Theta1 Theta2 D3 Theta4])
%Position (x,y,z) of the tool tip in zero position
Ptool_zero=HTzero_position*[0;0;0;1];
Ptool_zero=Ptool_zero(1:3);
Ptool_zero=double(Ptool_zero)
%Plot robot in zero position
figure(2)
SCARA.plot([Theta1 Theta2 D3 Theta4],'workspace',W,'scale', 2)


%% R8: Robot in offset position

disp('R8: Robot in offset position')
Theta1=pi/2
Theta2=-pi/2 
D3=-.125 
Theta4=pi
%End frame homogenous transformation matrix in offset position
HToffset_position = SCARA.fkine([Theta1 Theta2 D3 Theta4])
%Position (x,y,z) of the tool tip in offset position
Ptool_offset=HToffset_position*[0;0;0;1];
Ptool_offset=Ptool_offset(1:3);
Ptool_offset=double(Ptool_offset)
%Plot robot in offset position
SCARA.plot([Theta1 Theta2 D3 Theta4],'workspace',W,'scale', 2)
hold on


%% R9: Inverse kinematics for red and green pieces

disp('R9: Inverse kinematics for red and green pieces')
%Creating a target
%Green piece
greenp=transl(.6,.6,0)*trotz(pi/4);
trplot(greenp,'frame','1','color','g','length', 0.2)
hold on

%red piece
redp=transl(.3,.8,.1)*trotz(pi/2);
trplot(redp,'frame','2','color','r','length', 0.2)
hold on

%Solve inverse kinematic of green position
Qgreen=SCARA.ikine(greenp,[0,0,0,0],[1,1,1,0,0,1])
%check
SCARA.plot(Qgreen,'workspace',W,'scale', 2)

%Solve inverse kinematic of red position
Qred=SCARA.ikine(redp,[0,0,0,0],[1,1,1,0,0,1])
%check
SCARA.plot(Qred,'workspace',W,'scale', 2)


%% R10: Forward kinematics using fkine for the green work piece

disp('R10: Forward kinematics using fkine')
%HT matrix
HTgreenpiece_fkine = SCARA.fkine(Qgreen)
%location of the end effector
Ptool_fkine=HTgreenpiece_fkine*[0;0;0;1];
Ptool_fkine=Ptool_fkine(1:3);
Ptool_fkine=double(Ptool_fkine)



%% R11: Forward kinematics using symbolic toolbox

disp('R11: Forward kinematics using symbolic toolbox')
HTgreenpiece1=subs(TT0,[theta1 theta2 d3 theta4],Qgreen);
%TB1 add the desplacement of the base of the SCARA robot
TB1=[1 0 0 .1;0 1 0 .1;0 0 1 .36; 0 0 0 1];
HTgreenpiece2=TB1*HTgreenpiece1;
%HT matrix
HTgreenpiece2_symbolic_toolbox=double(HTgreenpiece2)
%location of the end effector
Ptool_symbolic_toolbox=HTgreenpiece2_symbolic_toolbox*[0;0;0;1];
Ptool_symbolic_toolbox=Ptool_symbolic_toolbox(1:3);
Ptool_symbolic_toolbox=double(Ptool_symbolic_toolbox)




