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

    @show (m_y, v_y) = unsafeMeanCov(y)
    @show m_kappa = x.params[:k]
    @show m_omega = x.params[:w]
    @show Psi = x.params[:psi]
    @show expansion_point = m_y + sqrt(v_y)*randn()
    @show phi = exp(m_kappa*expansion_point+m_omega)
    @show g = -0.5*((m_kappa*expansion_point+m_omega)+Psi/phi+ (expansion_point-m_y)^2/v_y)
    @show gprime = -0.5*(m_kappa -  Psi*m_kappa/phi +2*(expansion_point-m_y)/v_y)
    @show gdprime = -0.5*(m_kappa^2*Psi/phi+ 2/v_y)
    @show gddprime = -0.5*(-m_kappa^3*Psi/phi)
    @show var = -(1/gdprime)
    @show newton_update = -gprime/gdprime
    @show temp = 0.5*newton_update*(gddprime/gdprime)
    if  0.9 < abs(temp) < 1.1
        # println("Newton update")
        @show mean = expansion_point + newton_update
    else
        # println("Halley update")
        @show mean = expansion_point + newton_update*(1/(1+temp+1e-8))
    end

    z.params[:m] = mean
    z.params[:v] = var

    return z
end

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
