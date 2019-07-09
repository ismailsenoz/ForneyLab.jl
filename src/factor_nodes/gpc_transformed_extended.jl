export GPCLinearExtended

"""
Description:

    A Gaussian with covariance parameterized by exponential:

    f(out,m,v) = 𝒩(out|m,exp(kappa*v+omega)) = (2π)^{-1/2}exp(v)^{-1/2} exp(-1/2 (out - m)^2 /exp(kappa*v+omega))

Interfaces:

    1. out
    2. m
    3. v

Construction:

    GPC(out, m, v, id=:some_id)
"""

mutable struct GPCLinearExtended<: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GPCLinearExtended(out, m, v, kappa, omega; id=generateId(GPC))
        @ensureVariables(out, m, v, kappa, omega)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:v] = self.interfaces[3] = associate!(Interface(self), v)
        self.i[:kappa] = self.interfaces[4] = associate!(Interface(self),kappa)
        self.i[:omega] = self.interfaces[5] = associate!(Interface(self),omega)
        return self
    end
end

slug(::Type{GPCLinearExtended}) = "GPCLinear"

ProbabilityDistribution(::Type{Univariate}, ::Type{GPCLinearExtended}; m=0.0, v=1.0) = ProbabilityDistribution{Univariate, GaussianMeanPrecision}(Dict(:m=>m, :v=>exp(kappa*v+omega)))

#avarage energy functional for structured factorization


function averageEnergy(::Type{GPCLinearExtended}, marg_out_mean::ProbabilityDistribution{Multivariate}, marg_var::ProbabilityDistribution{Univariate},
                        marg_kappa::ProbabilityDistribution{Univariate, PointMass},marg_omega::ProbabilityDistribution{Univariate,PointMass})
    (m_out_mean, v_out_mean) = unsafeMeanCov(marg_out_mean)
    (m_var,v_var) = unsafeMeanCov(marg_var)
    m_kappa = marg_kappa.params[:m]
    m_omega = marg_omega.params[:m]
    gamma = exp(m_kappa*m_var+m_omega-(m_kappa^2)*v_var/2)

    0.5*log(2*pi) +
    0.5*(m_kappa*m_var+m_omega) +
    0.5*gamma*(v_out_mean[1,1]-v_out_mean[1,2]-v_out_mean[2,1]+v_out_mean[2,2]+ (m_out_mean[1]-m_out_mean[2])^2)
end

#avarage energy functional for mean field factorizatino
function averageEnergy(::Type{GPCLinearExtended}, marg_out::ProbabilityDistribution{Univariate},marg_mean::ProbabilityDistribution{Univariate},marg_var::ProbabilityDistribution{Univariate},
                        marg_kappa::ProbabilityDistribution{Univariate, PointMass},marg_omega::ProbabilityDistribution{Univariate,PointMass})
    (m_out, v_out) = unsafeMeanCov(marg_out)
    (m_mean, v_mean) = unsafeMeanCov(marg_mean)
    (m_var,v_var) = unsafeMeanCov(marg_var)
    m_kappa = marg_kappa.params[:m]
    m_omega = marg_omega.params[:m]
    gamma = v_mean + exp(m_kappa*m_var+m_omega)

    0.5*log(2*pi) +
    0.5*(m_kappa*m_var+m_omega) +
    0.5*1/gamma*(v_out+ (m_out-m_mean)^2)
end
