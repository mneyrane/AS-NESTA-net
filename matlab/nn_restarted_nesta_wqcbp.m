% Solve weighted QCBP (analysis) using restarted NESTA (assumes rows of 
% matrix A are pairwise orthogonal up to a shared constant c)
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
% in_iters (int) - Number of inner iterations.
% re_iters (int) - Number of outer iterations (restarts).
% eta (double) - Noise level defining the convex constraint set.
% mu0 (double) - Algorithm parameter (must be > 0).
% delta (double) - Algorithm parameter (estimate of scaled image error)
% r (double) - Decay parameter (must have 0 < r < 1).
% store_hist (logical) - Whether or not to store all the iterates computed 
%                        along the way.
%
% Returns
% -------
% x_final (vector) - Reconstructed coefficients.
% iterates (cell)  - If store_hist = 1, this is a cell array with all the
%                    iterates, otherwise it is an empty cell array
%
function [x_final,all_iterates] = nn_restarted_nesta_wqcbp(y, z0, opA, opW, L_W, in_iters, re_iters, eta, mu0, delta, r, store_hist)
    
    mu = mu0;
    z = z0;
    all_iterates = cell([re_iters,1]);
    
    for k=1:re_iters
        mu = r*(mu+delta);
        [z, cell_in_iterates] = nn_nesta_wqcbp(y, z, opA, opW, L_W, in_iters, eta, mu, store_hist);
        
        if (store_hist)
            all_iterates{k} = cell_in_iterates;
        end
    end
    x_final = z;
end
