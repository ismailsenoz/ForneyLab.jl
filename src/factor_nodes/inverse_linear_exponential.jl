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

    (m_y, v_y) = unsafeMeanCov(y)
    m_kappa = x.params[:k]
    m_omega = x.params[:w]
    Psi = x.params[:psi]
    expansion_point = m_y + sqrt(v_y)*randn()
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

    z.params[:m] = mean
    z.params[:v] = var

    return z
end

# using ForwardDiff
# import Distributions
#
# #differentiable reparameterization function for Gaussian
# function g_func(epsilon, a, b)
#     return a + b*epsilon
# end
#
# #approximating exact marginal posterior with gaussian q and auto-vi
# function gaussian_avi(log_fA, log_fB, mu_prior, v_prior, eta_a, eta_b, num_epoch, num_epoch2)
#     sum_a = 0
#     sum_b = 0
#     a_t = mu_prior
#     b_t = sqrt(v_prior)
#     for epoch2=1:num_epoch2
#         for epoch=1:num_epoch
#             message_6 = rand(Distributions.Normal(0,1),1)[1]
#             epsilon = message_6
#             message_7 = a_t
#             message_8 = b_t
#             message_9 = g_func(message_6, message_7, message_8)
#             x_sample = message_9
#             message_10 = x_sample
#             message_11 = x_sample
#             message_1 = ForwardDiff.derivative(log_fA, x_sample)
#             message_2 = ForwardDiff.derivative(log_fB, x_sample)
#             message_3 = message_1 + message_2
#             g_a(a) = g_func(epsilon, a, b_t)
#             message_4 = message_3 * ForwardDiff.derivative(g_a, a_t)
#             g_b(b) = g_func(epsilon, a_t, b)
#             message_5 = message_3 * ForwardDiff.derivative(g_b, b_t) + 1.0/b_t
#             sum_a += message_4
#             sum_b += message_5
#         end
#         a_t = a_t + eta_a*sum_a/num_epoch
#         b_t = b_t + eta_b*sum_b/num_epoch
#     end
#     return a_t, b_t^2
# end
#
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
#                             y::ProbabilityDistribution{Univariate, F},
#                             z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian
#
#     (m_y, v_y) = unsafeMeanCov(y)
#     m_kappa = x.params[:k]
#     m_omega = x.params[:w]
#     Psi = x.params[:psi]
#
#     fA(mu,v,s) = Distributions.pdf(Distributions.Normal(mu, v),s)
#     log_fB(omega,kappa,psi,s) = -0.5*(kappa*s+omega) - 0.5*(psi/(exp(kappa*s+omega)))
#     log_FA(s) = log(fA(m_y,v_y,s))
#     log_FB(s) = log_fB(m_omega,m_kappa,Psi,s)
#     (m_prior , v_prior) = (m_y, v_y)
#     mean, var = gaussian_avi(log_FA, log_FB, m_prior , v_prior , 0.0001, 0.0000001, 50, 500)
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end
#

#
#
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
#                             y::ProbabilityDistribution{Univariate, F},
#                             z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian
#
#     println("Inverse Linear Exp computations")
#     (m_y, v_y) = unsafeMeanCov(y)
#     m_kappa = x.params[:k]
#     m_omega = x.params[:w]
#     Psi = x.params[:psi]
#     expansion_point = m_y + sqrt(v_y)*randn()
#     phi = exp(m_kappa*expansion_point+m_omega)
#     g = -0.5*((m_kappa*expansion_point+m_omega)+Psi/phi+ (expansion_point-m_y)^2/v_y)
#     gprime = -0.5*(m_kappa -  Psi*m_kappa/phi +2*(expansion_point-m_y)/v_y)
#     gdprime = -0.5*(m_kappa^2*Psi/phi+ 2/v_y)
#     gddprime = -0.5*(-m_kappa^3*Psi/phi)
#     var = -(1/gdprime)
#     newton_update = -gprime/gdprime
#     temp = 0.5*newton_update*(gddprime/gdprime)
#     if  0.9 < abs(temp) < 1.1
#         # println("Newton update")
#         mean = expansion_point + newton_update
#     else
#         # println("Halley update")
#         mean = expansion_point + newton_update*(1/(1+temp+1e-8))
#     end
#     mean = expansion_point + newton_update*(1/(1+temp+1e-8))
#     temp = (1/(1+0.5*newton_update*gdprime/gprime))
#     mean = expansion_point + newton_update
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end

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
