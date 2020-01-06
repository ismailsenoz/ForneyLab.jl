using FastGaussQuadrature
using ForwardDiff, HCubature
export GaussianControlledVariance

"""
Description:

    A gaussian node where variance is controlled by a state that is passed
    through an exponential non-linearity.

    f(out,x,z,κ,ω) = N(out|x,exp(κz+ω))

Interfaces:

    1. out
    2. x (mean)
    3. z (state controlling the variance)
    4. κ
    5. ω

Construction:

    GaussianControlledVariance(out,x,z,κ,ω)
"""
mutable struct GaussianControlledVariance <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GaussianControlledVariance(out, x, z, κ, ω; id=generateId(GaussianControlledVariance))
        @ensureVariables(out, x, z, κ, ω)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)
        self.i[:κ] = self.interfaces[4] = associate!(Interface(self), κ)
        self.i[:ω] = self.interfaces[5] = associate!(Interface(self), ω)

        return self
    end
end

slug(::Type{GaussianControlledVariance}) = "GCV"


# Average energy functional
function averageEnergy(::Type{GaussianControlledVariance}, marg_out_x::ProbabilityDistribution{Multivariate}, marg_z::ProbabilityDistribution{Univariate}, marg_κ::ProbabilityDistribution{Univariate}, marg_ω::ProbabilityDistribution{Univariate})
    m_out_x, cov_out_x = unsafeMeanCov(marg_out_x)
    m_z, var_z = unsafeMeanCov(marg_z)
    m_κ, var_κ = unsafeMeanCov(marg_κ)
    m_ω, var_ω = unsafeMeanCov(marg_ω)

    ksi = (m_κ^2)*var_z + (m_z^2)*var_κ + var_κ*var_z
    psi = (m_out_x[2]-m_out_x[1])^2 + cov_out_x[1,1]+cov_out_x[2,2]-cov_out_x[1,2]-cov_out_x[2,1]
    A = exp(-m_ω + var_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    0.5log(2*pi) + 0.5*(m_z*m_κ+m_ω) + 0.5*(psi*A*B)
end
# Average energy functional
function averageEnergy(::Type{GaussianControlledVariance}, marg_out_x::ProbabilityDistribution{Multivariate}, marg_z_κ::ProbabilityDistribution{Multivariate}, marg_ω::ProbabilityDistribution{Univariate})
    m_out_x, cov_out_x = unsafeMeanCov(marg_out_x)
    m_z_κ, var_z_κ = unsafeMeanCov(marg_z_κ)
    m_ω, var_ω = unsafeMeanCov(marg_ω)

    psi = (m_out_x[2]-m_out_x[1])^2 + cov_out_x[1,1]+cov_out_x[2,2]-cov_out_x[1,2]-cov_out_x[2,1]
    A = exp(-m_ω + var_ω/2)
    B = quadratureExpectationExp(marg_z_κ,30)
    C = qudratureExpectationMultiplication(marg_z_κ,30)

    0.5log(2*pi) + 0.5*(C+m_ω) + 0.5*(psi*A*B)
end

function quadratureExpectationExp(d::ProbabilityDistribution{Multivariate,GaussianMeanVariance},p::Int64)
    sigma_points, sigma_weights = gausshermite(p)
    sigma_weights = sigma_weights
    m, v = unsafeMeanCov(d)
    result = 0.0
    for i=1:p
        result += sigma_weights[i]*exp((m[1]+sqrt(2*v[1,1])*sigma_points[i])*(m[2]+sqrt(2*v[2,2])*sigma_points[i]))/sqrt(pi)
    end

    return result
end

function quadratureExpectationMultiplication(d::ProbabilityDistribution{Multivariate,GaussianMeanVariance},p::Int64)
    sigma_points, sigma_weights = gausshermite(p)
    sigma_weights = sigma_weights
    m, v = unsafeMeanCov(d)
    result = 0.0
    for i=1:p
        result += sigma_weights[i]*(m[1]+sqrt(2*v[1,1])*sigma_points[i])*(m[2]+sqrt(2*v[2,2])*sigma_points[i])/sqrt(pi)
    end

    return result
end


function NewtonMethod(g::Function,x_0::Array{Float64},n_its::Int64)
    dim = length(x_0)
    x = zeros(dim)
    var = zeros(dim,dim)
    for i=1:n_its
        grad = ForwardDiff.gradient(g,x_0)
        hessian = ForwardDiff.hessian(g,x_0)
        x = x_0 - inv(hessian)*grad
        x_0 = x
    end
    var = inv(ForwardDiff.hessian(g,x))./2
    return x, var
end

function multivariateNormalApproximation(g::Function,x_min,x_max)

    normalization = HCubature.hcubature(g,x_min,x_max)[1]
    f(x) = x.* g(x)/normalization
    m = HCubature.hcubature(f,x_min,x_max)[1]
    h(x) = (x-m)*(x-m)'*g(x)/normalization
    v = HCubature.hcubature(h,x_min,x_max)[1]

    return Array(m), Array(v)
end
