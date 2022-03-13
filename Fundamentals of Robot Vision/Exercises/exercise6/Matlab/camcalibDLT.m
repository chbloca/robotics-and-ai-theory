function P=camcalibDLT(Xworld,Xim)
    % Inputs: 
    %   Xworld, world coordinates in the form (id, coordinates)
    %   Xim, image coordinates in the form (id, coordinates)
    
    % Create the matrix A 
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
    % Perform homogeneous least squares fitting,
    % the best solution is given by the eigenvector of 
    % A.T*A with the smallest eigenvalue.
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%

    % Reshape the eigenvector into a projection matrix P
    %P=(reshape(ev,4,3))'; % uncomment this
    P = [1 0 0 0;0 1 0 0;0 0 1 0];  % remove this

end