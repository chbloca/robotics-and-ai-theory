% Uncomment the commented code lines and implement missing functions 
% (and anything else that is asked or needed)

%% Load test images.
% Note: Must be double precision in the interval [0,1].
im = double(imread('einsteinpic.jpg'))/255;

%% Add noise 
% "salt and pepper" noise
imns = imnoise(im,'salt & pepper',0.1);
% zero-mean Gaussian noise
imng = im+0.05*randn(size(im));
imng(imng<0) = 0; imng(imng>1) = 1;

%% Display original and noise corrupted images
figure(1)
subplot(1,3,1)
imshow(im)
title('Original')
subplot(1,3,2)
imshow(imns)
title('Salt and Pepper')
subplot(1,3,3)
imshow(imng)
title('Gaussian noise')

%% Apply Gaussian filter of size 11x11 and std 2.5 
sigmad=2.5;
h = fspecial('gaussian', [11 11],2.5);
tmp_imns=imfilter(imns,h);
tmp_imng=imfilter(imng,h);

%% Instead of directly filtering with h, make a separable implementation
%  where you use horizontal and vertical 1D convolutions

% That is, replace the above two lines, you can use conv2 with two inputs instead
% The result should not change.

%%--your-code-starts-here--%%

[U, S, V] = svd(h);
hh = V(:,1)';
hv = S(1,1)*U(:,1);
gflt_imns = conv2(fliplr(hh), flipud(hv), imns, 'same');
gflt_imng = conv2(fliplr(hh), flipud(hv), imng, 'same');

%%--your-code-ends-here--%%


%% Apply median filtering, use neighborhood size 5x5
%  Open median_filter.m and implement the missing code
%  Store the results in medflt_imns and medflt_imng.
%  Use the median_filter.m function


%%--your-code-starts-here--%%
medflt_imns = median_filter(imns,5);
medflt_imng = median_filter(imng,5);
%%--your-code-ends-here--%%

%% Apply bilateral filter to each image with window size 11.
% See section 3.3.1 of Szeliski's book
% Use sigma value 2.5 for the domain kernel and 0.1 for range kernel.

% Set bilateral filter parameters.
w     = 0;       % bilateral filter half-width, filter size = 2*w+1 = 11
sigma = [0 0];   % sigma_d=sigma(1), sigma_r=sigma(2)

% Apply bilateral filter to each image.
bflt_imns = bilateralfilter(imns,w,sigma);
bflt_imng = bilateralfilter(imng,w,sigma);

%% Display grayscale input image and filtered output.
figure(2); clf;
set(gcf,'Name','Filtering Results');

subplot(2,4,1); imagesc(imns);
axis image; colormap gray;
title('Input Image');

subplot(2,4,2); imagesc(gflt_imns);
axis image; colormap gray;
title('Result of Gaussian Filtering');

subplot(2,4,3); imagesc(medflt_imns);
axis image; colormap gray;
title('Result of Median Filtering');

subplot(2,4,4); imagesc(bflt_imns);
axis image; colormap gray;
title('Result of Bilateral Filtering');

subplot(2,4,5); imagesc(imng);
axis image; colormap gray;
title('Input Image');

subplot(2,4,6); imagesc(gflt_imng);
axis image; colormap gray;
title('Result of Gaussian Filtering');

subplot(2,4,7); imagesc(medflt_imng);
axis image; colormap gray;
title('Result of Median Filtering');

subplot(2,4,8); imagesc(bflt_imng);
axis image; colormap gray;
title('Result of Bilateral Filtering');
