export GPC

"""
Description:

    A Gaussian with covariance parameterized by exponential:

    f(out,m,v) = 𝒩(out|m,exp(v)) = (2π)^{-1/2}exp(v)^{-1/2} exp(-1/2 (out - m)^2 /exp(v))

Interfaces:

    1. out
    2. m
    3. v

Construction:

    GPC(out, m, v, id=:some_id)
"""

mutable struct GPC<: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GPC(out, m, v; id=generateId(GPC))
        @ensureVariables(out, m, v)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:v] = self.interfaces[3] = associate!(Interface(self), v)

        return self
    end
end

slug(::Type{GPC}) = "GPC"

ProbabilityDistribution(::Type{Univariate}, ::Type{GPC}; m=0.0, v=1.0) = ProbabilityDistribution{Univariate, GaussianMeanPrecision}(Dict(:m=>m, :v=>exp(v)))

#avarage energy functional

function averageEnergy(::Type{GPC}, marg_out_mean::ProbabilityDistribution{Multivariate}, marg_var::ProbabilityDistribution{Univariate})
    (m_out_mean, v_out_mean) = unsafeMeanCov(marg_out_mean)
    (m_var,v_var) = unsafeMeanCov(marg_var)

    gamma = exp(-m_var+v_var/2)

    0.5*log(2*pi) -
    0.5*m_var +
    0.5*gamma*(v_out_mean[1]-v_out_mean[3]-v_out_mean[2]+v_out_mean[4]+ (m_out_mean[1]-m_out_mean[2])^2)
end
