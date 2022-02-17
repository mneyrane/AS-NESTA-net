% Demo for reconstructing an image from subsampled Fourier measurements.
% This is done by using restarted NESTA to solve an analysis QCBP problem
% where the analysis operator is the discrete gradient (or TV) operator.
%
% In this code, the measurements sample each frequency at most once to be
% compatible with NESTA. This results in a measurement matrix with
% orthonormal rows (up to a normalizing factor).
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
%X = phantom(512);


% parameters

eta = 1e-5;         % - noise level
[N, ~] = size(X);   % - image size (assumed to be N by N)
inner_iters = 6;    % - number of NESTA iterations
outer_iters = 5;    % - number of restarts
mu0 = 0.1;          % - initial smoothing parameter
delta = 1e-12;      % - ?
r = 0.5;            % - smoothing decay factor


% generate sampling mask 
% - i.e. a zero-one matrix with same shape as the image
% - can also load in a sampling mask from data or an image

sample_rate_target = 1.0;
m_target = ceil(sample_rate_target*N*N);

ber_opt_probs = h_bernoulli_opt_sampling_probs(N,m_target);

uniform_mask = rand(N,N) < m_target/N^2;
variable_mask = binornd(ones(N,N),ber_opt_probs);

mask = uniform_mask | variable_mask;
sample_idxs = find(mask);
m = length(sample_idxs);


% display info

disp(['Image size (number of pixels): ' num2str(N^2)])
disp(['Number of measurements: ' num2str(sum(mask(:)))])
disp(['Sample rate: ' num2str(sum(mask(:)) / N^2)])


% generate function handles for measurement and analysis operators

% measurement operator (subsampled Fourier transform)
subsampled_ft = @(x, mode) h_op_fourier_2d(x, mode, N, sample_idxs)*N/sqrt(m);

% discrete gradient (TV) operator
discrete_grad = @(x, mode) h_op_discrete_gradient_2d(x, mode, N, N);
L_grad = 2.0; % - Lipschitz constant of discrete_grad


% create measurement vector

noise = (eta/sqrt(m))*randn(m,1);
y = subsampled_ft(reshape(X, [N^2,1]),1) + noise;


% perform image reconstruction

z0 = zeros(N^2, 1);

[x_rec_vec, iterates] = nn_restarted_nesta_wqcbp(...
    y, z0, ...
    subsampled_ft, discrete_grad, L_grad, ...
    inner_iters, outer_iters, ...
    eta, mu0, delta, r, 1);

X_rec = reshape(real(x_rec_vec),[N,N]);


% error calculation

errs = zeros(outer_iters,1);
for i=1:outer_iters
    errs(i) = norm(X - reshape(iterates{i}{inner_iters+1},[N N]),'fro')/norm(X,'fro');
end


% PLOTS

figure(1)
imshow(mask)
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
