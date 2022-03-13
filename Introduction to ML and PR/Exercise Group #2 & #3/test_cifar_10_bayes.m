%% bayessian clasifier, naive approach

f_train = cifar_10_features(tr_data);
f_test = cifar_10_features(te_data);

[mu,sigma,p] =  cifar_10_bayes_learn(f_train,tr_labels);
c = zeros(100,1);
for i = 1:100
    c(i) = cifar_10_bayes_classify(f_test(i,:),mu,sigma,p);
end
c=c-1
accuracy=cifar_10_evaluate(c,labels)

%% bayessian classifier
    
f_train = cifar_10_features(tr_data);
f_test = cifar_10_features(te_data);

[mu,sigma,p] =  cifar_10_bayes_learn2(f_train,tr_labels);
for i = 1:100
    c(i) = cifar_10_bayes_classify2(f_test(i,:),mu,sigma,p);
end
c=c-1
accuracy=cifar_10_evaluate(c,labels)
%% bayessian classifier with extended features

