function [mu,sigma,p] = cifar_10_bayes_learn(f_train,label_train)
% mu=zeros(size(f_train));
% sigma=zeros(size(f_train));
% p=zeros(size(f_train));
for i = 1:10
    mu(i,:) = mean(f_train(label_train == i-1,:));
    sigma(i,:) = std(f_train(label_train == i-1,1:3));
    p(i) = nnz(label_train == i-1)/length(label_train);
end
end