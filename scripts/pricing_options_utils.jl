function make_quantile(dx)
    return x -> quantile(Normal(), dx/2 + x)
end

###########
# Helper function for loading Sobol matrices
function row_to_mat(row,b,m)
    return stack(first(digits(x, base = b, pad = m),m) for x in first(row,m))
end

function load_seq_mat(filename,b,m,s)
    df = CSV.read(filename, DataFrame, header = false, delim = ' ')
    K = Matrix(df[1:s,1:m])
    C = [row_to_mat(K[i,:],b,m) for i in 1:s]
    return C
end

function kernel(xa, sigma, T, S_init)
    return S_init * exp(-sigma*sigma/(2*T) + xa * sqrt(T))
end

function expected_value(XA_prod, sigma, T, S_init)
    N = size(XA_prod, 1)
    s = size(XA_prod, 2)
    return [1/N * sum(kernel(XA_prod[k,j], sigma, T, S_init) for k in 1:N) for j in 1:s]
end


function pay_off(ES,K)
    return mean(@. max(0.0, ES - K))
end