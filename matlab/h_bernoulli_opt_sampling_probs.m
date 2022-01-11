function probs = h_bernoulli_opt_sampling_probs(N,m)

    mag_fn = @(x) max(abs(x),1);
    q_omega_fn = @(w) max(mag_fn(w(1)), mag_fn(w(2)));
    Q = zeros(N);

    for i=-N/2+1:N/2
        for j=-N/2+1:N/2
            Q(i+N/2,j+N/2) = q_omega_fn([i j]);
        end
    end

    phi_C = @(t) phi(t,m,Q);

    [C,~] = h_bisfn(phi_C,0,1,1000,1e-12);
    disp(['Estimated value of C: ',num2str(C)])

    probs = zeros(N);

    for i=-N/2+1:N/2
        for j=-N/2+1:N/2
            probs(i+N/2,j+N/2) = min(m*C*Q(i+N/2,j+N/2)^(-2),1);
        end
    end
    
end

function out = phi(t, m, Q)

    out = 0;

    for i=1:size(Q,1)
        for j=1:size(Q,2)
            out = out + min(t*m*Q(i,j)^(-2),1);
        end
    end

    out = out - m;
    
end

