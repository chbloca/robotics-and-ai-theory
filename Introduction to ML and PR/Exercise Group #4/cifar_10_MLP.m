%f_train = cifar_10_features(tr_data);
%f_test = cifar_10_features(te_data);

%[mu,sigma,p] =  cifar_10_bayes_learn(f_train,tr_labels);
%c = zeros(100,1);
%for i = 1:100
%    c(i) = cifar_10_bayes_classify(f_test(i,:),mu,sigma,p);
%end
%c=c-1
%accuracy=cifar_10_evaluate(c,labels)

cifar_10_read_data;
f=cifar_10_features(tr_data);
ones = ones(size(tr_labels'));
x=double(tr_labels');
tr_labels_reshaped=full(ind2vec(x+ones))';
net = patternnet([50 50]);
net = train(net,f',tr_labels_reshaped');
%view(net)
y = sim(net,(double(f')));
perf = perform(net,tr_labels_reshaped',y);
%estlabel=cifar_10_MLP_test(f,net)