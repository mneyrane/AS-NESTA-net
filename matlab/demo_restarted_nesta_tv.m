% Test of nn_restarted_nesta_wqcbp.m.
%
% -- Maksym Neyra-Nesterenko
% -- mneyrane@sfu.ca

clear
close
clc

rng(0)


% load image

X = double(imread('../demo_images/phantom_brain_512.png'))/255;
%X = phantom(512);

% parameters

eta = 1e-5;
[N, ~] = size(X);
inner_iters = 10;
outer_iters = 15;
mu0 = 0.1;
delta = 1e-12;
r = 0.5;


% generate sampling mask
%mask = zeros(N,N,'logical');
%for i=-N/2+1:N/2
%    for j=-N/2+1:N/2
%        if (sqrt(i^2 + j^2) <= 0.25*(N/2))
%            mask(i+N/2,j+N/2) = 1;
%        end
%    end
%end
%mask = mask | (rand(N,N) < 0.1);
%sample_idxs = find(mask);
%m = length(sample_idxs);

% ==== DEBUG ====
m = ceil(0.075*N*N);

ber_opt_probs = h_bernoulli_opt_sampling_probs(N,m);

uniform_mask = rand(N,N) < m/N^2;
variable_mask = binornd(ones(N,N),ber_opt_probs);

mask = uniform_mask | variable_mask;
sample_idxs = find(mask);
m = length(sample_idxs);
% ===============


disp(['Image size (number of pixels): ' num2str(N^2)])
disp(['Number of measurements: ' num2str(sum(mask(:)))])
disp(['Sample rate: ' num2str(sum(mask(:)) / N^2)])


% generate function handles for measurement and weight operators

subsampled_ft = @(x, mode) h_op_fourier_2d(x, mode, N, sample_idxs)*N/sqrt(m);
discrete_grad = @(x, mode) h_op_discrete_gradient_2d(x, mode, N, N);
L_grad = 2.0;


% define inverse problem

noise = (eta/sqrt(m))*randn(m,1);
y = subsampled_ft(reshape(X, [N^2,1]),1) + noise;

% compute restarted NESTA solution

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


% plots

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
ylabel('Absolute error')
drawnow
