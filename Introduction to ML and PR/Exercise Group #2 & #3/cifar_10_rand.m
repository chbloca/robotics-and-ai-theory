function cifar_10_rand(x,gt)
aux=0;
random=randi([0 9],length(x));
for pt = 1:length(x)
    random(pt);
    if random(pt) == gt(pt)
        aux=aux+1;  
    end
end
accuracy=aux/length(x)
end