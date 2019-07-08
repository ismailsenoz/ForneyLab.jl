export InverseLinearExponential

"""
Description:
    InverseLinearExponential distribution type

    Real scalar parameters: k, w, psi

    f(out, k, w, psi) ∝ exp( -0.5(k*out + w + psi/exp(k*out + w) ) )
"""
abstract type InverseLinearExponential <: SoftFactor end

slug(::Type{InverseLinearExponential}) = "ILE"

format(dist::ProbabilityDistribution{Univariate, InverseLinearExponential}) = "$(slug(InverseLinearExponential))(k=$(format(dist.params[:k])), w=$(format(dist.params[:w])), psi=$(format(dist.params[:psi])))"

ProbabilityDistribution(::Type{Univariate}, ::Type{InverseLinearExponential}; k=1.0, w=0.0, psi=1.0) = ProbabilityDistribution{Univariate, InverseLinearExponential}(Dict(:k=>k, :w=>w, :psi=>psi))
ProbabilityDistribution(::Type{InverseLinearExponential}; k=1.0, w=0.0, psi=1.0) = ProbabilityDistribution{Univariate, InverseLinearExponential}(Dict(:k=>k, :w=>w, :psi=>psi))

dims(dist::ProbabilityDistribution{Univariate, InverseLinearExponential}) = 1

@symmetrical function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
                            y::ProbabilityDistribution{Univariate, F},
                            z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian

    println("Inverse Linear Exp computations")
    (m_y, v_y) = unsafeMeanCov(y)
    m_kappa = x.params[:k]
    m_omega = x.params[:w]
    Psi = x.params[:psi]
    expansion_point = m_y + sqrt(v_y)*randn()
    # expansion_point = m_y + randn()
    phi = exp(m_kappa*expansion_point+m_omega)
    g = -0.5*((m_kappa*expansion_point+m_omega)+Psi/phi+ (expansion_point-m_y)^2/v_y)
    gprime = -0.5*(m_kappa -  Psi*m_kappa/phi +2*(expansion_point-m_y)/v_y)
    gdprime = -0.5*(m_kappa^2*Psi/phi+ 2/v_y)
    gddprime = -0.5*(-m_kappa^3*Psi/phi)
    var = -(1/gdprime)
    newton_update = -gprime/gdprime
    temp = 0.5*newton_update*(gddprime/gdprime)
    if  0.9 < abs(temp) < 1.1
        # println("Newton update")
        mean = expansion_point + newton_update
    else
        # println("Halley update")
        mean = expansion_point + newton_update*(1/(1+temp+1e-8))
    end
    mean = expansion_point + newton_update*(1/(1+temp+1e-8))
    temp = (1/(1+0.5*newton_update*gdprime/gprime))
    mean = expansion_point + newton_update

    z.params[:m] = mean
    z.params[:v] = var

    return z
end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
#                             y::ProbabilityDistribution{Univariate, F},
#                             z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian
#
#     println("Inverse Linear Exp computations")
#     (m_y, v_y) = unsafeMeanCov(y)
#     m_kappa = x.params[:k]
#     m_omega = x.params[:w]
#     Psi = x.params[:psi]
#     @show expansion_point = m_y + 1e-9*randn()
#     @show phi = exp(m_kappa*expansion_point+m_omega)
#     @show g = -0.5*((m_kappa*expansion_point+m_omega)+Psi/phi+ (expansion_point-m_y)^2/v_y)
#     @show gprime = -0.5*(m_kappa -  Psi*m_kappa/phi +2*(expansion_point-m_y)/v_y)
#     @show gdprime = -0.5*(m_kappa^2*Psi/phi+ 2/v_y)
#     @show gddprime = -0.5*(-m_kappa^3*Psi/phi)
#     @show var = -(1/gdprime)
#     @show newton_update = -g/gprime
#     @show temp = 0.5*newton_update*(gdprime/gprime)
#     if  0.9 < abs(temp) < 1.1
#         println("Newton update")
#         @show mean = expansion_point + newton_update
#     else
#         println("Halley update")
#         @show mean = expansion_point + newton_update*(1/(1+temp+1e-8))
#     end
#     # @show mean = expansion_point + newton_update*(1/(1+temp+1e-8))
#     # @show temp = (1/(1+0.5*newton_update*gdprime/gprime))
#     # @show mean = expansion_point + newton_update
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end


# # Laplace
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
#                             y::ProbabilityDistribution{Univariate, F},
#                             z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian
#
#     println("Inverse Linear Exp computations")
#     (m_y, v_y) = unsafeMeanCov(y)
#     m_kappa = x.params[:k]
#     m_omega = x.params[:w]
#     Psi = x.params[:psi]
#     @show expansion_point = m_y
#     @show phi = exp(m_kappa*expansion_point+m_omega)
#     @show g = -0.5*(m_kappa - Psi*m_kappa/phi + 2*(expansion_point - m_y)/v_y)
#     @show gprime = -0.5*(m_kappa^2*Psi/phi+ 2/v_y)
#     @show gdprime = -0.5*(-m_kappa^3*Psi/phi)
#     @show var = -(1/gdprime)
#     @show newton_update = -g/gprime
#     @show temp = 0.5*newton_update*(gdprime/gprime)
#     if  0.9 < abs(temp) < 1.1
#         println("Newton update")
#         @show mean = expansion_point + newton_update
#     else
#         println("Halley update")
#         @show mean = expansion_point + newton_update*(1/(1+temp+1e-8))
#     end
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end
