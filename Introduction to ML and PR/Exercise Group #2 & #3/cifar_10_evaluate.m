%60000 images 32x32
%10 classes
%6000 image per class
%-5000 training images
%-1000 test images
%inputs: pred, gt
%- pred: predictions (1x10000)
%- gt: ground truth (1x10000)

function accuracy = cifar_10_evaluate(pred,gt)
aux=0;
for pt = 1:length(pred)
    if pred(pt) == gt(pt)
        aux=aux+1;  
    end
end
accuracy=aux/length(pred)
end
