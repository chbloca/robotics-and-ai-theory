function [ F ] = estimateF(x1,x2)
    % Input points are stacked column vectors, i.e. the row id corresponds
    % to the point id.
    
    % Use x1 and c2 to construct equation for homogeneous linear system
    % Each point correspondence gives a row vector described in the exercise sheet,
    % stack these to form a matrix used for estimating F.
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
    % Calculate SVD from our matrix above.
    % Extract fundamental matrix from the column of V corresponding to
    % smallest singular value.
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
    % Enforce constraint that fundamental matrix has rank 2 by performing
    % svd and then reconstructing with only the two largest singular values.
    % Reconstruction for matrix M:
    % M = U*S*V'
    % Where S is a diagonal matrix containing the singular values
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
end

