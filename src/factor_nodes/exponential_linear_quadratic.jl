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

vague(::Type{ExponentialLinearQuadratic}) = ProbabilityDistribution(Univariate, ExponentialLinearQuadratic, a=0.0, b=tiny, c=0.0, d=0.0)

ProbabilityDistribution(::Type{Univariate}, ::Type{ExponentialLinearQuadratic}; a=1.0,b=1.0,c=1.0,d=1.0)= ProbabilityDistribution{Univariate, ExponentialLinearQuadratic}(Dict(:a=>a, :b=>b, :c=>c, :d=>d))
ProbabilityDistribution(::Type{ExponentialLinearQuadratic}; a=1.0,b=1.0,c=1.0,d=1.0) = ProbabilityDistribution{Univariate, ExponentialLinearQuadratic}(Dict(:a=>a, :b=>b, :c=>c, :d=>d))

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff
using Cubature, HCubature

# function kldiv(d1::ProbabilityDistribution,d2::ProbabilityDistribution)
#     (m1,v1) = unsafeMeanCov(d1)
#     (m2,v2) = unsafeMeanCov(d2)
#
#     return 0.5*log(v2/v1) + (v1+(m1-m2)^2)/(2*v2) - 0.5
#
# end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate, F1},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     x_min = m_y-100.0
#     x_max = m_y+100
#
#     g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))*exp(-0.5*(x-m_y)^2/v_y)
#     normalization = hquadrature(g,x_min,x_max)[1]
#     f(x) = x.* g(x)./ normalization
#     m = hquadrature(f,x_min,x_max)[1]
#     h(x) = (x.- m).^2 .* g(x)./ normalization
#     v = hquadrature(h,x_min,x_max)[1]
#
#
#     z.params[:m] = m
#     z.params[:v] = v
#
#     return z
# end

function approximateDoubleExp(msg::Message{ExponentialLinearQuadratic})
    a = msg.dist.params[:a]
    b = msg.dist.params[:b]
    c = msg.dist.params[:c]
    d = msg.dist.params[:d]
    g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
    norm_const = HCubature.hquadrature(g,-100.0,100.0)[1]
    f(x) = x.* g(x)./ norm_const
    m = HCubature.hquadrature(f,-100.0,100.0)[1]
    h(x) = (x.- m).^2 .* g(x)./ norm_const
    v = HCubature.hquadrature(h,-100.0,100.0)[1]

    return Message(GaussianMeanVariance,m=m,v=v)
end

@symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
                y::ProbabilityDistribution{Univariate, F1},
                z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian

    dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
    m_y, v_y = unsafeMeanCov(dist_y)
    a = x.params[:a]
    b = x.params[:b]
    c = x.params[:c]
    d = x.params[:d]
    p = 20

    g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
    normalization_constant = quadrature(g,dist_y,p)
    t(x) = x*g(x)/normalization_constant
    mean_post = quadrature(t,dist_y,p)
    s(x) = (x-mean_post)^2*g(x)/normalization_constant
    var_post = quadrature(s,dist_y,p)


    z.params[:m] = mean_post
    z.params[:v] = var_post


    return z
end


# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate, F1},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#
#     g_der(x) = -0.5*(a + b*(c+d*x)*exp(c*x+d*x^2/2)+2*(x-m_y)/v_y)
#     g_hes(x) = -0.5*(exp(c*x+d*x^2/2)*(b*d+b*(c+d*x)^2)+ 2/v_y)
#     mean_post = m_y
#     for i=1:100
#         @show mean_post = mean_post - g_der(mean_post)/g_hes(mean_post)
#     end
#     # println("derivative ", g_der(mean_post)," ","hessian ", g_hes(mean_post))
#
#     @show z.params[:m] = mean_post
#     @show z.params[:v] = -2/g_hes(mean_post)
#
#
#     return z
# end


#
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y = dist_y.params[:m]
#     v_y = dist_y.params[:v]
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     g(x) = -0.5*(a*x+b*exp(c*x+d*x^2/2)+(x-m_y)^2/v_y)
#     h(x) = ForwardDiff.derivative(g,x)
#     # var_z = 0.0
#     # mean_z = 0.0
#     # for i=1:100
#     gradient = ForwardDiff.derivative(g, m_y)
#     hessian = ForwardDiff.derivative(h, m_y)
#     var_z = -inv(hessian)
#     mean_z = m_y + var_z*gradient
#     m_y = mean_z
#     # end
#     println(mean_z, " ", var_z)
#     z.params[:m] = mean_z
#     z.params[:v] = var_z
#
#     return z
# end

#
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y = dist_y.params[:m]
#     v_y = dist_y.params[:v]
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     f(x) = exp(c*x+d*x^2/2)
#     g_der(x) = -0.5*(a + b*f(x)*(c+d*x)+2*(x-m_y)/v_y)
#     g_hes(x) = -0.5*(f(x)*b*(a^2-2*a*x*d+x^2*d+d) + 2/v_y)
#     # mean_z = 0.0
#     # for i = 1:10
#     println(g_der(m_y), g_hes(m_y))
#     mean_z = 0.0
#     for i=1:100
#         mean_z = m_y - 0.1*g_der(m_y)
#         m_y = mean_z
#     end
#     # end
#     z.params[:m] = mean_z
#     z.params[:v] = -1/g_hes(mean_z)
#
#
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
    m, v = ForneyLab.unsafeMeanCov(d)
    result = 0.0
    for i=1:p
        result += sigma_weights[i]*g(m+sqrt(2*v)*sigma_points[i])/sqrt(pi)
    end
    return result
end

function laplaceApproximation(dist::ProbabilityDistribution{Univariate,ExponentialLinearQuadratic},x_0,dim::Int64)
    a = dist.params[:a]
    b = dist.params[:b]
    c = dist.params[:c]
    d = dist.params[:d]
    g(x) = -0.5*(a*x+b*exp(c*x+d*x^2/2))
    mean = 0.0
    var = 0.0
    for i=1:dim
        grad = ForwardDiff.derivative(g,x_0)
        hessian = ForwardDiff.derivative(x -> ForwardDiff.derivative(g,x),x_0)
        mean = x_0 - inv(hessian)*grad
        var = -0.5*inv(hessian)
        x_0 = mean
    end
    return ProbabilityDistribution(GaussianMeanVariance,m=mean,v=var)
end
