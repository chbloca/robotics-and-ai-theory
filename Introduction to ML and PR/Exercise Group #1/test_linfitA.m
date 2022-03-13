figure
axis([0 5 0 5]);
[x,y]=ginput(5);

plot(x,y, 'o')
[a,b]=linfitA(x,y);
y_val=a*x+b;
hold on
plot(x,y_val, '-')