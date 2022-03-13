function [ estimation ] = cifar_10_1NN( x, trdata, trlabels )

x_rep = repmat(x, length(trdata), 1);

euclidean_distance = sum((x_rep - trdata).*(x_rep - trdata), 2);

[~, index] = min(euclidean_distance);

estimation = trlabels(index);

end

