export
ruleSVBGaussianControlledVarianceOutNGDDD,
ruleSVBGaussianControlledVarianceXGNDDD,
ruleSVBGaussianControlledVarianceZDNDD,
ruleSVBGaussianControlledVarianceΚDDND,
ruleSVBGaussianControlledVarianceΩDDDN,
ruleMGaussianControlledVarianceGGDDD

# function sigmaPointsAndWeights(dist::ProbabilityDistribution{Univariate, F}, alpha::Float64) where F<:Gaussian
#     (m_x, V_x) = unsafeMeanCov(dist)
#
#     kappa = 0
#     beta = 2
#     lambda = (1 + kappa)*alpha^2 - 1
#
#     sigma_points = Vector{Float64}(undef, 3)
#     weights_m = Vector{Float64}(undef, 3)
#     weights_c = Vector{Float64}(undef, 3)
#
#     l = sqrt((1 + lambda)*V_x)
#
#     sigma_points[1] = m_x
#     sigma_points[2] = m_x + l
#     sigma_points[3] = m_x - l
#     weights_m[1] = lambda/(1 + lambda)
#     weights_m[2] = weights_m[3] = 1/(2*(1 + lambda))
#     weights_c[1] = weights_m[1] + (1 - alpha^2 + beta)
#     weights_c[2] = weights_c[3] = 1/(2*(1 + lambda))
#
#     return (sigma_points, weights_m, weights_c)
# end


function ruleSVBGaussianControlledVarianceOutNGDDD(dist_out::Nothing,
                                                   msg_x::Message{F, Univariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_x.dist)
    m_x = dist_x.params[:m]
    v_x = dist_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, GaussianMeanVariance, m=m_x, v=v_x+inv(A*B))
end

function ruleSVBGaussianControlledVarianceXGNDDD(msg_out::Message{F, Univariate},
                                                   dist_x::Nothing,
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_y = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_y.dist)
    m_y = dist_y.params[:m]
    v_y = dist_y.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, GaussianMeanVariance, m=m_y, v=v_y+inv(A*B))
end

function ruleSVBGaussianControlledVarianceZDNDD(dist_out_x::ProbabilityDistribution{F, Multivariate},
                                                   dist_z::Nothing,
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, ExponentialLinearQuadratic, a=m_κ, b=Psi*A,c=-m_κ,d=v_κ)
end

function ruleSVBGaussianControlledVarianceΚDDND(dist_out_x::ProbabilityDistribution{F, Multivariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::Nothing,
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, ExponentialLinearQuadratic, a=m_z, b=Psi*A,c=-m_z,d=v_z)
end

function ruleSVBGaussianControlledVarianceΩDDDN(dist_out_x::ProbabilityDistribution{F, Multivariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::Nothing) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, ExponentialLinearQuadratic, a=1.0, b=Psi*B,c=-1.0,d=0.0)
end

function ruleMGaussianControlledVarianceGGDDD(msg_out::Message{F1, Univariate},
                                              msg_x::Message{F2, Univariate},
                                              dist_z::ProbabilityDistribution{Univariate},
                                              dist_κ::ProbabilityDistribution{Univariate},
                                              dist_ω::ProbabilityDistribution{Univariate})
    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_out.dist)
    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_x.dist)
    m_x = dist_x.params[:m]
    w_x = dist_x.params[:w]
    m_out = dist_out.params[:m]
    w_out = dist_out.params[:w]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)
    W = [w_out+A*B -A*B; -A*B w_x+A*B]
    m = inv(W)*[m_out*w_out; m_x*w_x]

    return ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m, w=W)

end
