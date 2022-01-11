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
 

% parameters

eta = 1e-5;
[N, ~] = size(X);
m = ceil(0.075*N*N);
inner_iters = 12;
outer_iters = 15;
mu0 = 0.1;
delta = 1e-12;
r = 0.5;


% generate sampling masks

ber_opt_probs = h_bernoulli_opt_sampling_probs(N,m);

uniform_mask = rand(N,N) < m/N^2;
variable_mask = binornd(ones(N,N),ber_opt_probs);

uniform_idxs = find(uniform_mask);
variable_idxs = find(variable_mask);

m1 = length(uniform_idxs);
m2 = length(variable_idxs);

disp(['Image size (number of pixels): ' num2str(N^2)])
disp(['Number of measurements: ' num2str(m1+m2)])
disp(['Parameter m = ', num2str(m)])
disp(['Sample rate: ' num2str((m1+m2)/N^2)])


% define inverse problem

noise = (eta/sqrt(m1+m2))*randn(m1+m2,1);
vec_X = reshape(X,[N^2 1]);
u1 = h_op_fourier_2d(vec_X,1,N,uniform_idxs)*N/sqrt(m) + noise(1:m1);
u2 = h_op_fourier_2d(vec_X,1,N,variable_idxs)*N/sqrt(m) + noise(m1+1:m1+m2);


% generate function handles for measurement and weight operators

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

subsampled_ft = @(x, mode) h_op_fourier_2d(x, mode, N, sample_idxs)*N/sqrt(m);
discrete_grad = @(x, mode) h_op_discrete_gradient_2d(x, mode, N, N);
L_grad = 2.0;


% compute restarted NESTA solution

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
ylabel('Absolute error')
drawnow
