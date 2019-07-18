export
ruleVBGPCLinearExtendedOutVGGPP,
ruleVBGPCLinearExtendedMeanGVGPP,
ruleVBGPCLinearExtendedVarGGVPP,
ruleSVBGPCLinearExtendedOutVGGPP,
ruleSVBGPCLinearExtendedMeanGVGPP,
ruleSVBGPCLinearExtendedVarGVPP,
ruleMGPCLinearExtendedGGDDD


function ruleVBGPCLinearExtendedOutVGGPP(msg_out::Nothing,
                                dist_mean::ProbabilityDistribution{Univariate},
                                dist_v::ProbabilityDistribution{Univariate},
                                dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                                dist_omega::ProbabilityDistribution{Univariate,PointMass})

    (mean_m, cov_m) = unsafeMeanCov(dist_mean)
    (mean_v, cov_v) = unsafeMeanCov(dist_v)
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    gamma = exp(m_kappa*mean_v + m_omega)
    mean = mean_m
    var = cov_m + gamma

    Message(GaussianMeanVariance,m=mean,v=var)
end

function ruleVBGPCLinearExtendedMeanGVGPP(dist_out::ProbabilityDistribution{Univariate},
                                msg_mean::Nothing,
                                dist_v::ProbabilityDistribution{Univariate},
                                dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                                dist_omega::ProbabilityDistribution{Univariate,PointMass})

    (mean_out, cov_out) = unsafeMeanCov(dist_out)
    (mean_v, cov_v) = unsafeMeanCov(dist_v)
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    gamma = exp(m_kappa*mean_v + m_omega)
    mean = mean_out
    var = cov_out + gamma
    Message(GaussianMeanVariance,m=mean,v=var)

end

function ruleVBGPCLinearExtendedVarGGVPP(dist_out::ProbabilityDistribution{Univariate},
                                dist_mean::ProbabilityDistribution{Univariate},
                                msg_v::Nothing,
                                dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                                dist_omega::ProbabilityDistribution{Univariate,PointMass})
    (mean_out, cov_out) = unsafeMeanCov(dist_out)
    (mean_m, cov_m) = unsafeMeanCov(dist_mean)
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    Psi = cov_out+(mean_out-mean_m)^2

    Message(LogLinearExponential, v=cov_m, k=m_kappa, w=m_omega, psi=Psi)

end

function ruleSVBGPCLinearExtendedOutVGGPP(msg_out::Nothing,
                            msg_mean::Message{F,Univariate},
                            dist_v::ProbabilityDistribution{Univariate},
                            dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                            dist_omega::ProbabilityDistribution{Univariate,PointMass}) where F<:Gaussian
    d_mean = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance}, msg_mean.dist)
    m_v =  unsafeMean(dist_v)
    v_v = unsafeCov(dist_v)
    m_mean = d_mean.params[:m]
    v_mean = d_mean.params[:v]
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    gamma = exp(m_kappa*m_v +m_omega - (m_kappa^2)*v_v/2 )
    Message(GaussianMeanVariance, m=m_mean,  v=v_mean + gamma)
end

function ruleSVBGPCLinearExtendedMeanGVGPP(msg_out::Message{F,Univariate},
                            msg_mean::Nothing,
                            dist_v::ProbabilityDistribution,
                            dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                            dist_omega::ProbabilityDistribution{Univariate,PointMass}) where F<:Gaussian

    d_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
    m_out = d_out.params[:m]
    v_out = d_out.params[:v]
    m_v =  unsafeMean(dist_v)
    v_v = unsafeCov(dist_v)
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    gamma = exp(m_kappa*m_v + m_omega - (m_kappa^2)*v_v/2 )
    Message(GaussianMeanVariance, m=m_out,  v=v_out + gamma)
end

function ruleSVBGPCLinearExtendedVarGVPP(dist_out_mean::ProbabilityDistribution{Multivariate},
                            dist_v::Nothing,
                            dist_kappa::ProbabilityDistribution{Univariate},
                            dist_omega::ProbabilityDistribution{Univariate})


    (m, V) = unsafeMeanCov(dist_out_mean)
    m_kappa = unsafeMean(dist_kappa)
    m_omega = unsafeMean(dist_omega)
    Psi = V[1,1]-V[1,2]-V[2,1]+V[2,2]+(m[1]-m[2])^2

    Message(InverseLinearExponential, k=m_kappa, w=m_omega, psi=Psi)
end


function ruleMGPCLinearExtendedGGDDD(msg_out::Message{F1, Univariate},
                        msg_m::Message{F2, Univariate},
                        dist_v::ProbabilityDistribution{Univariate, F3},
                        dist_kappa::ProbabilityDistribution{Univariate},
                        dist_omega::ProbabilityDistribution{Univariate}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
    d_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
    d_mean = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_m.dist)

    m_mean = d_mean.params[:m]
    v_mean = d_mean.params[:v]
    m_out = d_out.params[:m]
    v_out = d_out.params[:v]
    (m_v, v_v) =  unsafeMeanCov(dist_v)
    m_kappa = unsafeMean(dist_kappa)
    m_omega = unsafeMean(dist_omega)

    gamma = exp(m_kappa*m_v - (m_kappa^2)*v_v/2 + m_omega)
    V = pinv([1/gamma + 1/v_out -1/gamma; -1/gamma 1/gamma + 1/v_mean])
    m = V*[m_mean*v_mean; m_out*v_out]

    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m, v=V)
end

###Working example for weighted mean precision updates
# function ruleMGPCLinearExtendedGGDDD(msg_out::Message{F1, Univariate},
#                         msg_m::Message{F2, Univariate},
#                         dist_v::ProbabilityDistribution{Univariate, F3},
#                         dist_kappa::ProbabilityDistribution{Univariate},
#                         dist_omega::ProbabilityDistribution{Univariate}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
#     println("Marginal calculation for GPC")
#     d_out = convert(ProbabilityDistribution{Univariate,GaussianWeightedMeanPrecision},msg_out.dist)
#     d_mean = convert(ProbabilityDistribution{Univariate,GaussianWeightedMeanPrecision},msg_m.dist)
#
#     xi_mean = d_mean.params[:xi]
#     w_mean = d_mean.params[:w]
#     xi_out = d_out.params[:xi]
#     w_out = d_out.params[:w]
#     (m_v, v_v) =  unsafeMeanCov(dist_v)
#     m_kappa = unsafeMean(dist_kappa)
#     m_omega = unsafeMean(dist_omega)
#
#     @show gamma = exp(-m_kappa*m_v + (m_kappa^2)*v_v/2 - m_omega)
#     # gamma = exp(-m_kappa*m_v - m_omega)
#     @show q_W = [w_out+(gamma) -gamma; -gamma w_mean+(gamma)]
#
#
#     q_xi = [xi_out; xi_mean]
#     return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=q_xi, w=q_W)
# end
