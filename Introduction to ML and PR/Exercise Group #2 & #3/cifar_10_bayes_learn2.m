function [mu,sigma,p] = cifar_10_bayes_learn2(f_train,label_train)
    for i = 1:10
        mu(i,:) = mean(f_train(label_train == i-1,1:3));
        sigma(:,:,i) = cov(f_train(label_train == i-1,1:3));
        p = nnz(label_train == i-1)/length(label_train);
    end
end