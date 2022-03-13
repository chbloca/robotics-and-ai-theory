%% Random classifier

K = length(data); % Number of images
pred_labels = zeros(K, 1);

for k = 1:K
    data_sample = data(k, :);
    pred_label = cifar_10_rand(data_sample);
    pred_labels(k) = pred_label;
end

cifar_10_evaluate(pred_labels, labels)

%% 1-NN classifier
tic;
K = length(data); % Number of images
K = 10000;
pred_labels = zeros(K, 1);

for k = 1:K
    data_sample = data(k, :);
    pred_label = cifar_10_1NN(data_sample, tr_data, tr_labels);
    pred_labels(k) = pred_label;
end

cifar_10_evaluate(pred_labels, labels(1:K))
toc;