
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
    readlines(f, keep = true)
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
display(cname)

using Statistics
occursin.(r"log_lik", cname)
sum(x -> occursin(r"log_lik", x), cname)
function wbic(smpl, cname)
    n = sum(x -> occursin(r"log_lik", x), cname)
    loc = findall(occursin.(r"log_lik", cname))
    log_lik_smpl = smpl[:,loc,:]
    sum(log_lik_smpl; dims = 2) |> mean
end

wbic(smpl, cname) |> display
# -308.65735830750003
## WBIC - v_t

function gf_variance(smpl, cname)
    n = sum(x -> occursin(r"log_lik", x), cname)
    loc = findall(occursin.(r"log_lik", cname))
    log_lik_smpl = smpl[:,loc,:]
    log_lik = sum(log_lik_smpl; dims = 2)
    mean(mean(log_lik_smpl .^2; dims = 2) .- mean(log_lik_smpl; dims = 2) .^2)
end
gf_variance(smpl, cname)

v_t = 0.5log(n) * gf_variance(smpl, cname)
wbic(smpl, cname) - v_t |> display
