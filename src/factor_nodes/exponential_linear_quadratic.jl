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

# @symmetric function prod!(x::ProbabilityDistribution{Univariate, Bernoulli},
#                 y::ProbabilityDistribution{Univariate, Bernoulli},
#                 z::ProbabilityDistribution{Univariate, Bernoulli}=ProbabilityDistribution(Univariate, Bernoulli, p=0.5))
#
#     norm = x.params[:p] * y.params[:p] + (1 - x.params[:p]) * (1 - y.params[:p])
#     (norm > 0) || error("Product of $(x) and $(y) cannot be normalized")
#     z.params[:p] = (x.params[:p] * y.params[:p]) / norm
#
#     return z
# end
