export ExponentialLinearQuadratic

"""
Description:



    f(out,a,b,c,d) = exp(-0.5(a*out + b*exp(cx+dx^2/2)))

Interfaces:

    1. out
    2. a
    3. b
    4. c
    5. d


"""
mutable struct ExponentialLinearQuadratic<: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function ExponentialLinearQuadratic(out, a, b, c, d; id=generateId(ExponentialLinearQuadratic))
        @ensureVariables(out, a, b, c, d)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:a] = self.interfaces[2] = associate!(Interface(self), a)
        self.i[:b] = self.interfaces[3] = associate!(Interface(self), b)
        self.i[:c] = self.interfaces[4] = associate!(Interface(self), c)
        self.i[:d] = self.interfaces[5] = associate!(Interface(self), d)

        return self
    end
end

slug(::Type{ExponentialLinearQuadratic}) = "ELQ"

format(dist::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic}) = "$(slug(ExponentialLinearQuadratic))(a=$(format(dist.params[:a])), b=$(format(dist.params[:b])), c=$(format(dist.params[:c])), d=$(format(dist.params[:d])))"

ProbabilityDistribution(::Type{Univariate}, ::Type{ExponentialLinearQuadratic}; a=1.0,b=1.0,c=1.0,d=1.0)= ProbabilityDistribution{Univariate, ExponentialLinearQuadratic}(Dict(:a=>a, :b=>b, :c=>c, :d=>d))
ProbabilityDistribution(::Type{ExponentialLinearQuadratic}; a=1.0,b=1.0,c=1.0,d=1.0) = ProbabilityDistribution{Univariate, ExponentialLinearQuadratic}(Dict(:a=>a, :b=>b, :c=>c, :d=>d))

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff

@symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
                y::ProbabilityDistribution{Univariate, F1},
                z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian

    dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
    m_y, v_y = unsafeMeanCov(dist_y)
    a = x.params[:a]
    b = x.params[:b]
    c = x.params[:c]
    d = x.params[:d]
    p = 5

    g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
    normalization_constant = quadrature(g,dist_y,p)
    t(x) = x*g(x)/normalization_constant
    mean = quadrature(t,dist_y,p)
    s(x) = (x-mean)^2*g(x)/normalization_constant
    var = quadrature(s,dist_y,p)

    z.params[:m] = mean
    z.params[:v] = var

    return z
end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     epsilon = 0.1
#     g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)+(x-m_y)^2/v_y))
#     h(x) = ForwardDiff.derivative(g,x)
#
#     for i=1:100
#         gradient = ForwardDiff.derivative(g, m_y)
#         hessian = ForwardDiff.derivative(h, m_y)
#         var = -inv(hessian)
#         mean = m_y + epsilon*var*gradient
#         m_y = mean
#     end
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F <: Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#
#     g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
#
#     samples = m_y .+ sqrt(v_y) .* randn(1000)
#     mean = sum(g.(samples) ./ sum(g.(samples)) .* samples)
#     var = sum(g.(samples) ./ sum(g.(samples)) .* (samples.-mean).^2)
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end


function quadrature(g::Function,d::ProbabilityDistribution{Univariate,GaussianMeanVariance},p::Int64)
    sigma_points, sigma_weights = gausshermite(p)
    sigma_weights = sigma_weights./ (sqrt(pi)*2^(p-1))
    m, v = unsafeMeanCov(d)
    result = 0.0
    for i=1:p
        result += sigma_weights[i]*g(m+sqrt(v)*sigma_points[i])
    end

    return result
end
