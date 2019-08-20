export MultivariateGPC
"""
Description:

    Gaussian with Parametrized Covariance in multivariate form. It represents a probability distribution
    N(out|m,g(v)) where g(v) is a mapping from d-dimensional real space to space of
    positive definite matrices.


Interfaces:

    1. out
    2. m
    3. v

Construction:

    MultivariateGPC(out, m, v, g::Function)
"""
mutable struct MultivariateGPC <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}
    g::Function
    dims::Tuple

    function MultivariateGPC(out, m, v, g::Function; dims=(1,), id=generateId(MultivariateGPC))
        @ensureVariables(out, m, v)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}(), g,dims)
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:v] = self.interfaces[3] = associate!(Interface(self), v)
        return self
    end
end

slug(::Type{MultivariateGPC}) = "GPC"
