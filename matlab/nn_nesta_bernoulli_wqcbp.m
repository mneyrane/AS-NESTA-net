% Solve weighted QCBP (analysis) where measurements correspond to a
% Bernoulli model. This uses an algorithm similar to NESTA derived from
% Nesterov's method. We assume that the measurement matrix A is of the form
% A = [B ; B] where the rows of B are pairwise orthogonal up to a shared 
% constant c). Note that the measurements in each vector MUST be aligned.
%
% Arguments
% ---------
% y1 (vector) - First measurement vector.
% y2 (vector) - Second measurement vector.
% z0 (vector) - Initial guess of x. 
% opB (function handle) - The sampling operator. opA(x,1) is the forward 
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
% iterates (cell)  - If store_hist = 1, this is a cell array with all the 
%                    iterates, otherwise it is an empty cell array
%
function [x_final,all_iterates] = nn_nesta_bernoulli_wqcbp(y1, y2, z0, opB, opW, L_W, num_iters, eta, mu, store_hist)

c = norm(opB(double(1:length(y1) == 1),0))^2; % -- could make this optional or an input --

z = z0;
q_v = z0;

all_iterates = cell([num_iters+1,1]);


% these quantities only need to be calculated once
% -------------------------------------------------
y_sum = y1+y2;
d_noise = 2*eta*eta-dot(y1-y2,y1-y2);
% -------------------------------------------------

for n=0:num_iters
    % -----------
    % compute x_n
    % -----------
    grad = opW(z,1);
    grad = h_op_huber_fn_gradient(grad,mu);
    grad = mu/(L_W*L_W)*opW(grad,0);
    q = z-grad;
    
    dy = y_sum-2*opB(q,1);
    lam = max(0,0.5*(1-sqrt(d_noise/dot(dy,dy))));
    
    x = (lam/c)*opB(dy,0) + q;
    
    if (store_hist)
        all_iterates{n+1} = x;
    end
    
    % -----------
    % compute v_n
    % -----------
    alpha = (n+1)/2;
    q_v = q_v-alpha*grad;
    q = q_v;
    
    dy = y_sum-2*opB(q,1);
    lam = max(0,0.5*(1-sqrt(d_noise/dot(dy,dy))));
    
    v = (lam/c)*opB(dy,0) + q;

    % ---------------
    % compute z_{n+1}
    % ---------------
    tau = 2/(n+3);
    z = tau*v+(1-tau)*x;
end

x_final = x;

end
