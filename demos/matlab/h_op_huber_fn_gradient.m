function out = h_op_huber_fn_gradient(x, mu)
    
out = zeros(size(x));
for i=1:length(x)
    if (abs(x(i)) <= mu)
        out(i) = x(i)/mu;
    else
        out(i) = x(i)/abs(x(i));
    end
end

end