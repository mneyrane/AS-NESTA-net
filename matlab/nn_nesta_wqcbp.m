% Solve weighted QCBP (analysis) using NESTA (assumes rows of matrix A are 
% pairwise orthogonal up to a shared constant c)
%
% Arguments
% ---------
% y (vector) - Measurement vector.
% z0 (vector) - Initial guess of x. 
% opA (function handle) - The sampling operator. opA(x,1) is the forward 
%                         transform, and opA(y,0) is the adjoint. We assume
%                         the forward map times the adjoint yields the
%                         identity map, and that the inputs and outputs are
%                         vectors.
% opW (function handle) - The analysis matrix W in the weighted QCBP
%                         problem. opW(x,1) is the forward map, and
%                         opW(y,0) is the adjoint. We assume the inputs and
%                         outputs are vectors.
% L_W (double) - Operator 2-norm of opW.
% num_iters (int) - Number of inner iterations.
% eta (double) - Noise level defining the convex constraint set.
% mu (double) - Smoothing parameter.
% store_hist (logical) - Whether or not to store all the iterates computed 
%                        along the way.
%
% Returns
% -------
% x_final (vector) - Reconstructed coefficients.  
% all_iterates (cell)  - If store_hist = 1, this is a cell array with all the 
%                    iterates, otherwise it is an empty cell array
%
function [x_final,all_iterates] = nn_nesta_wqcbp(y, z0, opA, opW, L_W, num_iters, eta, mu, store_hist)

c = norm(opA(double(1:length(y) == 1),0),2)^2; % -- could make this optional or an input --

z = z0;
q_v = z0;

all_iterates = cell([num_iters+1,1]);

for n=0:num_iters
    % -----------
    % compute x_n
    % -----------
    grad = opW(z,1);
    grad = h_op_huber_fn_gradient(grad,mu);
    grad = mu/(L_W*L_W)*opW(grad,0);
    q = z-grad;
    
    dy = y-opA(q,1);
    lam = max(0,norm(dy,2)/eta - 1);
    
    x = lam/((lam+1)*c)*opA(dy,0) + q;
    
    if (store_hist)
        all_iterates{n+1} = x;
    end
    
    % -----------
    % compute v_n
    % -----------
    alpha = (n+1)/2;
    q_v = q_v-alpha*grad;
    q = q_v;
    
    dy = y-opA(q,1);
    lam = max(0,norm(dy,2)/eta - 1);
    
    v = lam/((lam+1)*c)*opA(dy,0) + q;

    % ---------------
    % compute z_{n+1}
    % ---------------
    tau = 2/(n+3);
    z = tau*v+(1-tau)*x;
end

x_final = x;

end
