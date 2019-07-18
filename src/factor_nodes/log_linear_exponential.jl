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
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, LogLinearExponential},
#                             y::ProbabilityDistribution{Univariate, F},
#                             z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where F<:Gaussian
#
#     (m_y, v_y) = unsafeMeanCov(y)
#     m_kappa = x.params[:k]
#     m_omega = x.params[:w]
#     Psi = x.params[:psi]
#     v_x = x.params[:v]
#
#     fA(mu,v,s) = Distributions.pdf(Distributions.Normal(mu, v),s)
#     log_fB(omega,kappa,psi,w,s) = -0.5*log(w+exp(kappa*s+omega)) - 0.5*(psi/(w+exp(kappa*s+omega)))
#     log_FA(s) = log(fA(m_y,v_y,s))
#     log_FB(s) = log_fB(m_omega,m_kappa,Psi,v_x,s)
#     (m_prior , v_prior) = (m_y, v_y)
#     mean, var = gaussian_avi(log_FA, log_FB, m_prior , v_prior, 0.0001, 0.0000001, 50, 500)
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end
