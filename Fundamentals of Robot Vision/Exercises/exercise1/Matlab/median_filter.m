
function output = median_filter(A,wsize)
% Median filtering is done by extracting a local patch from the input image
% and calculating its median

% Apply median filter.
dim = size(A);
output = zeros(dim);

k = (wsize - 1) / 2;

for i = 1:dim(1)
   for j = 1:dim(2)
      
         % Extract local region.
         iMin = max(i-k,1);
         iMax = min(i+k,dim(1));
         jMin = max(j-k,1);
         jMax = min(j+k,dim(2));
         img_patch = A(iMin:iMax,jMin:jMax);
         
         % Calculate the median value from the extracted 
         % local region and store it to output using correct indexing.
         
         %%--your-code-starts-here--%%
         
         %%--your-code-ends-here--%%
               
   end
end
