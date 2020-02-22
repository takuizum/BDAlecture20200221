
using Pkg
Pkg.add(["CSV", "SpecialFunctions", "RCall"])

using LinearAlgebra, SpecialFunctions, RCall

R"""
# install.packages("RcppSMC")
data(radiata,package = "RcppSMC")
n <- nrow(radiata)
X <- cbind(rep(1,n),radiata$x1)
y <- matrix(radiata$y,ncol=1)
"""
n = X = y = nothing # Dummies for Linting
@rget n X y

r_0 = 0.06
s_0 = 6
Q_0 = [r_0 0;0 s_0]
a_0 = 6
b_0 = 600^2

## the exact value of marginal likelihood

M = X' * X + Q_0
R = I(n) - X * inv(M) * X'
ML = π^(-n/2) * b_0^(a_0/2) * (gamma((n+a_0)/2) / gamma(a_0/2)) *
  (det(Q_0)^(1/2) / det(M)^(1/2)) * ((y'*R*y)[1,1]+b_0)^(-(n+a_0)/2)
log(ML)

logML = -(n/2)*log(pi) + (a_0/2)*log(b_0) + log( gamma((n+a_0)/2)/gamma(a_0/2) ) +
  (1/2)*log( det(Q_0)/det(M) ) - ((n+a_0)/2)*log((y'*R*y)[1,1]+b_0)
logML

## Stan
using CmdStan
Imai2019_stan = open("Imai2019_model1.stan", "r") do f
    readlines(f; keep = true)
end |> join
mod1 = Stanmodel(name = "imai2019"; model = Imai2019_stan,
                 num_samples = 1000, num_warmup = 1000, thin = 1, nchains = 4, random = CmdStan.Random(1234))
display(mod1)

data1 = Dict("n" => n,
             "x" => X[:,2],
             "y" => y[:],
             "mode" => 1 # WBIC
             )

rc, smpl, cname = stan(mod1, data1)

using MCMCChains, DataFrames
summarize(smpl)

# Extract log_lik
function extract_loglik(smpl, cname)
    n_smpl = size(smpl)[1]
    nm = cname[occursin.(r"log_lik", cname)]
    tmp_smpl = get_params(smpl)
    chns = MCMCChains.chains(smpl)
    res = zeros(Float64, n_smpl*length(chns), length(nm))
    for (i, n) in enumerate(nm)
        sym = Symbol(n)
        res[:, i] = vec(tmp_smpl[sym])[:]
    end
    res = DataFrame(res)
    rename!(res, nm)
    return res
end
# extract_loglik(smpl, cname)

# Define `sum` for dataframe type.
function Base.sum(df::DataFrame; dims )
    sum(convert(Array, df), dims = dims)
end

using Statistics
function wbic(smpl, cname)
    smpl_loglik = extract_loglik(smpl, cname)
    sum(smpl_loglik; dims = 2) |> mean
end
wbic(smpl, cname) |> display
# -308.74247859999997

## WBIC - v_t

Statistics.mean(df::DataFrame; dims) = mean(convert(Array, df), dims = dims)
function gf_variance(smpl, cname)
    smpl_loglik = extract_loglik(smpl, cname)
    sum(mean(smpl_loglik .^2; dims = 1) .- mean(smpl_loglik; dims = 1) .^2)
end
gf_variance(smpl, cname)

v_t = (1/(2*log(n))) * gf_variance(smpl, cname)
wbic(smpl, cname) - v_t |> display
# -310.86545070746405
