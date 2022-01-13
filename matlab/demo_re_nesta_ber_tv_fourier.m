% Demo for reconstructing an image from subsampled Fourier measurements
% under the Bernoulli model.
%
% Here a restarted NESTA-like algorithm solves an analysis QCBP problem
% where the analysis operator is the discrete gradient (or TV) operator.
%
% In this code, the measurements are generated by sampling frequencies 
% using two Bernoulli masks, one corresponding to uniform sampling and
% the other variable density sampling. As a result, a frequency can be
% possibly (and at most) twice.
%
% -- Maksym Neyra-Nesterenko
% -- mneyrane@sfu.ca

clear
close
clc

% fix the RNG seed for debugging
%rng(0)


% load image

X = double(imread('../demo_images/phantom_brain_512.png'))/255;
 

% parameters

eta = 1e-5;             % - noise level
[N, ~] = size(X);       % - image size (assumed to be N by N)
m = ceil(0.25*N*N);     % - expected num of measurements per mask
inner_iters = 12;       % - number of NESTA iterations
outer_iters = 10;       % - number of restarts
mu0 = 1e-2;             % - initial smoothing parameter
delta = 1e-12;          % - ?
r = 0.5;                % - smoothing decay factor


% generate sampling masks
% - i.e. each mask is a zero-one matrix with same shape as the image
% - can also load in a sampling mask from data or an image

% -- NOTE --
% In this experiment, the masks are generated with respect to the
% Bernoulli model. The exact number of measurements need not be m.
ber_opt_probs = h_bernoulli_opt_sampling_probs(N,m);

uniform_mask = rand(N,N) < m/N^2;
variable_mask = binornd(ones(N,N),ber_opt_probs);

uniform_idxs = find(uniform_mask);
variable_idxs = find(variable_mask);

m1 = length(uniform_idxs);
m2 = length(variable_idxs);


% display some info

disp(['Image size (number of pixels): ' num2str(N^2)])
disp(['Exact number of measurements: ' num2str(m1+m2)])
disp(['Bernoulli model parameter m = ', num2str(m)])
disp(['Effective sample rate: ' num2str((m1+m2)/(2*N^2))])


% define measurement vector

noise = (eta/sqrt(m1+m2))*randn(m1+m2,1);
vec_X = reshape(X,[N^2 1]);
u1 = h_op_fourier_2d(vec_X,1,N,uniform_idxs)*N/sqrt(m) + noise(1:m1);
u2 = h_op_fourier_2d(vec_X,1,N,variable_idxs)*N/sqrt(m) + noise(m1+1:m1+m2);

omega2diff1 = setdiff(variable_idxs,uniform_idxs);
omega1 = [uniform_idxs ; omega2diff1];
u2_mask = ismember(variable_idxs,omega2diff1);
y1 = [u1 ; u2(u2_mask)];

omega1diff2 = setdiff(uniform_idxs,variable_idxs);
omega2 = [variable_idxs ; omega1diff2];
u1_mask = ismember(uniform_idxs,omega1diff2);
y2 = [u2 ; u1(u1_mask)];

[o1,perm1] = sort(omega1);
[o2,perm2] = sort(omega2);

y1 = y1(perm1);
y2 = y2(perm2);

if (~all(o1 == o2)) % -- DEBUG --
    error('Incorrectly implemented frequency sample stacking')
end

sample_idxs = o1;


% generate function handles for measurement and analysis operators

% measurement operator (subsampled Fourier transform)
subsampled_ft = @(x, mode) h_op_fourier_2d(x, mode, N, sample_idxs)*N/sqrt(m);

% discrete gradient (TV) operator
discrete_grad = @(x, mode) h_op_discrete_gradient_2d(x, mode, N, N);
L_grad = 2.0; % - Lipschitz constant of discrete_grad


% perform image reconstruction

z0 = zeros(N^2, 1);

[x_rec_vec, iterates] = nn_restarted_nesta_bernoulli_wqcbp(...
    y1, y2, z0, ...
    subsampled_ft, discrete_grad, L_grad, ...
    inner_iters, outer_iters, ...
    eta, mu0, delta, r, 1);

X_rec = reshape(real(x_rec_vec),[N,N]);


% error calculation

errs = zeros(outer_iters,1);
for i=1:outer_iters
    errs(i) = norm(X - reshape(iterates{i}{inner_iters+1},[N N]),'fro')/norm(X,'fro');
end


% plots

figure(1)
imshow(uniform_mask | variable_mask)
title('Sampling pattern')
drawnow

figure(2)
imshow(X)
title('Ground truth image')
drawnow

figure(3)
imshow(X_rec)
title('Reconstructed image')
drawnow

figure(4)
semilogy(1:outer_iters,errs,'*b')
xlabel('Restart')
ylabel('Relative error')
drawnow