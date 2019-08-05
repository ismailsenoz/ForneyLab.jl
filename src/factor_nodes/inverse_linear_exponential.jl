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
    expansion_point = m_y
    phi = exp(m_kappa*expansion_point+m_omega)
    g = -0.5*((m_kappa*expansion_point+m_omega)+Psi/phi+ (expansion_point-m_y)^2/v_y)
    gprime = -0.5*(m_kappa -  Psi*m_kappa/phi + 2*(expansion_point-m_y)/v_y)
    gdprime = -0.5*(m_kappa^2*Psi/phi+ 2/v_y)
    gddprime = -0.5*(-m_kappa^3*Psi/phi)
    var = -(1/gdprime)
    @show newton_update = -gprime/gdprime
    temp = 0.5*newton_update*(gddprime/gdprime)
    if  0.9 < abs(temp) < 1.1
        # println("Newton update")
        mean = expansion_point + newton_update
    else
        # println("Halley update")
        mean = expansion_point + newton_update*(1/(1+temp+1e-8))
    end
    # mean = expansion_point

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
#     log_fB(omega,kappa,psi,s) = -0.5*(kappa*s+omega + (psi/(exp(kappa*s+omega))))
#     log_FA(s) = log(fA(m_y,v_y,s))
#     log_FB(s) = log_fB(m_omega,m_kappa,Psi,s)
#     (m_prior , v_prior) = (m_y, v_y)
#     mean, var = gaussian_avi(log_FA, log_FB, m_prior , v_prior, 0.00001, 0.0000001, 50, 500)
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
#     (m_y, v_y) = unsafeMeanCov(y)
#     m_kappa = x.params[:k]
#     m_omega = x.params[:w]
#     @show Psi = x.params[:psi]
#     mean_approx = (log(Psi)-m_omega)/m_kappa
#     var_approx = 2/m_kappa^2
#     var = 1/(1/v_y + 1/var_approx)
#     mean = var*(m_y/v_y + mean_approx/var_approx)
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end
# function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
#                             y::ProbabilityDistribution{Univariate, InverseLinearExponential},
#                             z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0))
#
#
#     @show m_kappa_x = x.params[:k]
#     @show m_omega_x = x.params[:w]
#     @show Psi_x = x.params[:psi]
#     @show m_kappa_y = y.params[:k]
#     @show m_omega_y = y.params[:w]
#     @show Psi_y = y.params[:psi]
#
#     @show mean = (log(m_kappa_x*Psi_x)+log(m_kappa_y*Psi_y)-log(m_kappa_x+m_kappa_y))/(m_kappa_x+m_kappa_y)
#     @show phi_x = exp(-m_kappa_x*Psi_x-m_omega_x) + tiny
#     @show phi_y = exp(-m_kappa_y*Psi_y-m_omega_y) + tiny
#     @show var = 1/((m_kappa_x^2*Psi_x)/phi_x + (m_kappa_y^2*Psi_y)/phi_y)
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end

using ForwardDiff

function ile(kappa,omega,psi)
    function ile(x)
        phi = exp(kappa*x+omega)
        return  -0.5*(kappa*x+omega+psi/phi)
    end
    return ile
end

function gradILE(kappa,omega,psi)
    function gradILE(x)
        phi = exp(kappa*x+omega)
        return -0.5*kappa*(1-psi/phi)
    end
    return gradILE
end

function hessianILE(kappa,omega,psi)
    function hessianILE(x)
        phi = exp(kappa*x + omega)
        return -0.5*(kappa^2*psi/phi)
    end
    return hessianILE
end


function prod!(x::ProbabilityDistribution{Univariate, InverseLinearExponential},
                            y::ProbabilityDistribution{Univariate, InverseLinearExponential},
                            z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0))


    m_kappa_x = x.params[:k]
    m_omega_x = x.params[:w]
    Psi_x = x.params[:psi]
    m_kappa_y = y.params[:k]
    m_omega_y = y.params[:w]
    Psi_y = y.params[:psi]

    f_x = ile(m_kappa_x,m_omega_x,Psi_x)
    f_y = ile(m_kappa_y,m_omega_y,Psi_y)
    grad_f_x = gradILE(m_kappa_x,m_omega_x,Psi_x)
    grad_f_y = gradILE(m_kappa_y,m_omega_y,Psi_y)
    hessian_f_x = hessianILE(m_kappa_x,m_omega_x,Psi_x)
    hessian_f_y = hessianILE(m_kappa_y,m_omega_y,Psi_y)
    initial_x = 1.0
    mean = initial_x - (grad_f_x(initial_x)+grad_f_y(initial_x))/(hessian_f_x(initial_x)+hessian_f_y(initial_x))
    var = -1/(hessian_f_x(mean)+hessian_f_y(mean))
    z.params[:m] = mean
    z.params[:v] = var

    return z
end
