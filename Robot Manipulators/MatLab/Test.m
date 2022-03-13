syms alpha a d theta;

%From frame i to frame i-1
Ti2prev = trotx(alpha)*transl([a 0 0])*transl([0 0 d])*trotz(theta)
Ti2prev = simplify(Ti2prev)

%% 
%theta2=90

T21 = subs(Ti2prev, [alpha, a, d, theta],  [0, .75, 0, 0])
T21 = double(T21)

%theta1=0
T10 = subs(Ti2prev, [alpha, a, d, theta],  [0, 0, 0, 0])
T10 = double(T10)

T20 = T10*T21

%%
L(1)=Link([0, 0, 0, 0], 'modified') %theta, d, a, alpha
L(2)=Link([0, 0, 0.75, 0], 'modified')

two_link=SerialLink(L, 'name', 'two_link')

%offsent is for defining the initial pose

%%

two_link.fkine([0, pi/2])

%%

two_link.plot([0, 0])

%%

two_link.teach([0, 0])

%%

ToolHT = transl([0.5, 0, 0])
two_link.tool = ToolHT

%%

BaseHT = transl([0, 0.3, 0])
two_link.base = BaseHT

%%

two_link.plot([0,0])

%plotting
hold on
trplot(target,'frame','T','color','r')