function f = cifar_10_features(x)
k=length(x);
f_r=zeros(k,1);
f_g=zeros(k,1);
f_b=zeros(k,1);
f=zeros(k,3);


for i=1:k
    f_r(i) = mean(x(i,1:1024));
    
    f_g(i) = mean(x(i,(1024+1):2*1024));
    
    f_b(i) = mean(x(i,2*(1024+1):3*1024));
    
    f(i,1:3) = [f_r(i),f_g(i),f_b(i)];
    
end
end