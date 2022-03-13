function [a,b] = linfitA(x,y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

b = (sum(y).*sum(x.^2)-sum(x).*sum(x.*y))/(5*(sum(x.^2))-(sum(x)^2));
a =  (sum(x.*y)-b*sum(x))/sum(x.^2);
end

