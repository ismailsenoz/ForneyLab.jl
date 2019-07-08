export LogLinearExponential

"""
Description:
    LogLinearExponential distribution type


"""
abstract type LogLinearExponential <: SoftFactor end

slug(::Type{LogLinearExponential}) = "LLE"

format(dist::ProbabilityDistribution{Univariate,LogLinearExponential}) = "$(slug(LogLinearExponential))(v=$(format(dist.params[:v])),k=$(format(dist.params[:k])), w=$(format(dist.params[:w])), psi=$(format(dist.params[:psi])))"

ProbabilityDistribution(::Type{Univariate}, ::Type{LogLinearExponential}; v=1.0, k=1.0, w=0.0, psi=1.0) = ProbabilityDistribution{Univariate, LogLinearExponential}(Dict(:v=>v, :k=>k, :w=>w, :psi=>psi))
ProbabilityDistribution(::Type{LogLinearExponential}; v=1.0, k=1.0, w=0.0, psi=1.0) = ProbabilityDistribution{Univariate, LogLinearExponential}(Dict(:v=>v, :k=>k, :w=>w, :psi=>psi))

dims(dist::ProbabilityDistribution{Univariate, LogLinearExponential}) = 1

@symmetrical function prod!(x::ProbabilityDistribution{Univariate, LogLinearExponential},
                            y::ProbabilityDistribution{Univariate, F},
                            z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian

    (m_y, v_y) = unsafeMeanCov(y)
    m_kappa = x.params[:k]
    m_omega = x.params[:w]
    Psi = x.params[:psi]
    v = x.params[:v]
    phi = exp(m_kappa*m_y+m_omega)
    expansion_point = m_y
    temp = v + phi
    gprime = -0.5*(m_kappa*phi/(temp) -  Psi*m_kappa*phi/(temp)^2)
    gdprime = -0.5*((temp*m_kappa^2*phi-m_kappa^2*phi^2)/temp^2 - (temp^2*Psi*m_kappa^2*phi-2*Psi*m_kappa^2*phi^2*temp)/temp^4 + 2/v_y)
    var = -(1/gdprime)
    newton_update = -gprime/gdprime

    mean = expansion_point + newton_update

    z.params[:m] = mean
    z.params[:v] = var

    return z
end
