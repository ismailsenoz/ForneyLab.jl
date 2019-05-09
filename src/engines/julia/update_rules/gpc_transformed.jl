export
ruleVBGPCLinearOutVPPPP,
ruleSVBGPCLinearOutVGGPP,
ruleSVBGPCLinearMeanGVGPP,
ruleSVBGPCLinearVarGVPP,
ruleMGPCLinearGGDDD


function ruleVBGPCLinearOutVPPPP(msg_out::Nothing,
                            dist_mean::ProbabilityDistribution{Univariate, PointMass},
                            dist_v::ProbabilityDistribution{Univariate, PointMass},
                            dist_kappa::ProbabilityDistribution{Univariate, PointMass},
                            dist_omega::ProbabilityDistribution{Univariate, PointMass})
    (m_mean) = dist_mean.params[:m]
    (m_v) = dist_v.params[:m]
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    gamma = exp(m_kappa*m_v+m_omega)

    Message(GaussianMeanVariance, m=m_mean,  v=gamma)
end

function ruleSVBGPCLinearOutVGGPP(  msg_out::Nothing,
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
    gamma = exp(m_kappa*m_v - (m_kappa^2)*v_v/2 + m_omega)

    Message(GaussianMeanVariance, m=clamp(m_mean,tiny,huge),  v=clamp(v_mean + gamma,tiny,huge))
end

function ruleSVBGPCLinearMeanGVGPP( msg_out::Message{F,Univariate},
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
    gamma = exp(m_kappa*m_v - (m_kappa^2)*v_v/2 + m_omega)
    Message(GaussianMeanVariance, m=clamp(m_out,tiny,huge),  v=clamp(v_out + gamma,tiny,huge))
end

function ruleSVBGPCLinearVarGVPP(   dist_out_mean::ProbabilityDistribution{Multivariate},
                            dist_v::Nothing,
                            dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                            dist_omega::ProbabilityDistribution{Univariate,PointMass})

    (m, V) = unsafeMeanCov(dist_out_mean)
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    A = V[1]-V[2]-V[3]+V[4]+(m[1]-m[2])^2
    mean = (log(A)-m_omega)/m_kappa

    Message(GaussianMeanVariance, m=clamp(mean,tiny,huge),  v=clamp(2/(m_kappa^2*A^2),tiny,huge))

end

function ruleMGPCLinearGGDDD(   msg_out::Message{F1, Univariate},
                        msg_m::Message{F2, Univariate},
                        dist_v::ProbabilityDistribution{Univariate, F3},
                        dist_kappa::ProbabilityDistribution{Univariate,PointMass},
                        dist_omega::ProbabilityDistribution{Univariate,PointMass}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}

    d_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
    d_mean = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_m.dist)
    d_v = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},dist_v)

    m_mean = d_mean.params[:m]
    v_mean = d_mean.params[:v]
    w_mean = 1/v_mean
    m_out = d_out.params[:m]
    v_out = d_out.params[:v]
    w_out = 1/v_out
    m_v =  unsafeMean(dist_v)
    v_v = unsafeCov(dist_v)
    w_v = 1/v_v
    m_kappa = dist_kappa.params[:m]
    m_omega = dist_omega.params[:m]
    gamma = exp(-m_kappa*m_v + (m_kappa^2)*v_v/2 - m_omega)
    determinant = 1/(w_out*w_mean + gamma*(w_out+w_mean))

    invW = determinant .* [w_mean+gamma gamma; gamma w_out+gamma]
    mean = invW*[w_out*m_out; w_mean*m_mean]
    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=mean, v=invW)
end
