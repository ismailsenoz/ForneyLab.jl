export LogDetTrace


mutable struct LogDetTrace <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}
    g::Function
    dims::Tuple
    function LogDetTrace(out,Psi,g::Function,cov,;dims=(1,0), id=generateId(LogDetTrace))
        @ensureVariables(out, Psi,cov)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:Psi] = self.interfaces[2] = associate!(Interface(self), Psi)
        self.i[:cov] = self.interfaces[3] = associate!(Interface(self), cov)
        return self
    end
end
slug(::Type{LogDetTrace}) = "LDT"

format(dist::ProbabilityDistribution{Multivariate,LogDetTrace}) = "$(slug(LogDetTrace))(Psi=$(format(dist.params[:Psi])))"

ProbabilityDistribution(::Type{Multivariate}, ::Type{LogDetTrace}; g::Function, Psi=[1.0 0.0; 0.0 1.0], cov=[1.0 0.0; 0.0 1.0])= ProbabilityDistribution{Multivariate, LogDetTrace}(Dict(:Psi=>Psi, :g=>g, :cov=>cov))
ProbabilityDistribution(::Type{LogDetTrace}; g::Function, Psi=[1.0 0.0; 0.0 1.0],cov=[1.0 0.0; 0.0 1.0]) = ProbabilityDistribution{Multivariate, LogDetTrace}(Dict(:Psi=>Psi, :g=>g,:cov=>cov))

dims(dist::ProbabilityDistribution{Multivariate, LogDetTrace}) = dims

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff

@symmetrical function prod!(x::ProbabilityDistribution{Multivariate, LogDetTrace},
                            y::ProbabilityDistribution{Multivariate, F},
                            z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=[0.0;0.0], v=[1.0 0.0; 0.0 1.0])) where F<:Gaussian


    Psi = x.params[:Psi]
    g = x.params[:g]
    cov = x.params[:cov]
    (m_y, cov_y) = ForneyLab.unsafeMeanCov(y)
    dim = dims(y)

    G(z::AbstractArray{<:Real}) = -0.5*(logdet(g(z)) + LinearAlgebra.tr(inv(g(z))*Psi))
    gradient = ForwardDiff.gradient(G, m_y)
    hessian = ForwardDiff.hessian(G, m_y)
    var = -inv(hessian)
    mean = m_y + var*gradient
    z.params[:m] = mean
    z.params[:v] = var


    return z
end


#
# @symmetrical function prod!(x::ProbabilityDistribution{Multivariate, LogDetTrace},
#                             y::ProbabilityDistribution{Multivariate, F},
#                             z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=[0.0;0.0], v=[1.0 0.0; 0.0 1.0])) where F<:Gaussian
#
#
#     Psi = x.params[:Psi]
#     g = x.params[:g]
#     cov = x.params[:cov]
#     convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, y)
#     dim = dims(y)
#     points_iter, weights_iter = generate_multidim_points(dim, 5)
#     G(z::AbstractArray{<:Real}) = exp(-0.5*(logdet(g(z)) + LinearAlgebra.tr(inv(g(z))*Psi)))
#     normalization_constant = gaussHermiteCubature(G, y, points_iter, weights_iter)
#     t(z::AbstractArray{<:Real}) = z
#     mean = gaussHermiteCubatureMean(t,y,points_iter,weights_iter)
#     h(z::AbstractArray{<:Real}) = (z-mean)*transpose((z-mean))
#     cov = gaussHermiteCubatureCov(h,y,points_iter,weights_iter)
#     z.params[:m] = mean
#     z.params[:v] = cov
#
#     println(mean, " ", cov)
#     println(normalization_constant)
#
#     return z
# end


const product = Iterators.product
const PIterator = Iterators.ProductIterator

function generate_multidim_points(n::Int, p::Int)
    sigma_points, sigma_weights = gausshermite(p)
    points_iter = product(repeat([sigma_points],n)...)
    weights_iter = product(repeat([sigma_weights],n)...)
    return points_iter, weights_iter
end

function gaussHermiteCubature(g::Function, d::ProbabilityDistribution{Multivariate,GaussianMeanVariance}, points_iter::PIterator, weights_iter::PIterator)
    result = 0.0
    sqrtP = sqrt(d.params[:v])
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        result += weight.*g(d.params[:m] + sqrtP*point)
    end

    return result
end

function gaussHermiteCubatureMean(g::Function, d::ProbabilityDistribution{Multivariate,GaussianMeanVariance}, points_iter::PIterator, weights_iter::PIterator)
    result = zeros(dims(d))
    sqrtP = sqrt(d.params[:v])
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        result = result + weight.*g(d.params[:m] + sqrtP*point)
    end

    return result
end

function gaussHermiteCubatureCov(g::Function, d::ProbabilityDistribution{Multivariate,GaussianMeanVariance}, points_iter::PIterator, weights_iter::PIterator)
    result = zeros(dims(d),dims(d))
    sqrtP = sqrt(d.params[:v])
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        result = result + weight.*g(d.params[:m] + sqrtP*point)
    end

    return result
end









# using FastGaussQuadrature
# using Einsum
# using Combinatorics
#
#
# function multidimensionalWeightsNodes(p::Int64,dims::Int64)
#     (nodes, weights) = gausshermite(p)
#     weights_matrix = repeat(weights, outer=[1, dims])
#     @einsum W[i,j,k] := weights[i,1]*weights[j,1]*weights_matrix[1,k]
#     nodes_matrix = collect(combinations(nodes,dims))
#     return W
# end
#
#
# function mdNodes(nodes::Array{Float64},n::Int64)
#     m = length(nodes)
#     nodes_repeat = repeat(nodes)
#     permutes = collect(permutations(nodes_repeat))
#     combs = Array{Array{Array{Float64,1},1}}(undef,m)
#     for i=1:m
#         combs[i] = collect(combinations(permutes[i],n))
#     end
#     return combs
# end
#



# function generateHermite(p::Int64)
#     Hermite = Function[]
#
#     push!(Hermite,x->1);
#     push!(Hermite,x->x);
#
#     for i=3:p
#         push!(Hermite,x->x.*Hermite[i-1](x).-p.*Hermite[i-2](x))
#     end
#     return Hermite
# end
