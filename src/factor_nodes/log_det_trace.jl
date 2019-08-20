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

using LinearAlgebra
using ForwardDiff

@symmetrical function prod!(x::ProbabilityDistribution{Multivariate, LogDetTrace},
                            y::ProbabilityDistribution{Multivariate, F},
                            z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=[0.0;0.0], v=[1.0 0.0; 0.0 1.0])) where F<:Gaussian

    Psi = x.params[:Psi]
    g = x.params[:g]
    cov = x.params[:cov]
    (m_y, cov_y) = ForneyLab.unsafeMeanCov(y)

    G(z::AbstractArray{<:Real}) = -logdet(g(z)) - LinearAlgebra.tr(inv(g(z))*Psi)
    gradient = ForwardDiff.gradient(G, m_y)
    hessian = ForwardDiff.hessian(G,m_y)
    var = -inv(hessian)
    mean = m_y + var*gradient
    z.params[:m] = mean
    z.params[:v] = var

    return z
end
