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
