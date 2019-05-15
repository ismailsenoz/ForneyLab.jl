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
    gamma = exp(m_kappa*m_v + (m_kappa^2)*v_v/2 + m_omega)
    println("forward messages")
    println(m_mean," ", v_mean, " ", gamma)

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
    gamma = exp(m_kappa*m_v + (m_kappa^2)*v_v/2 + m_omega)
    println("backward messages")
    println(m_out," ", v_out, " ", gamma)
    Message(GaussianMeanVariance, m=clamp(m_out,tiny,huge),  v=clamp(v_out + gamma,tiny,huge))
end

function ruleSVBGPCLinearVarGVPP(   dist_out_mean::ProbabilityDistribution{Multivariate},
                            dist_v::Nothing,
                            dist_kappa::ProbabilityDistribution{Univariate},
                            dist_omega::ProbabilityDistribution{Univariate})

    (m, V) = unsafeMeanCov(dist_out_mean)
    m_kappa = unsafeMean(dist_kappa)
    m_omega = unsafeMean(dist_omega)
    A = V[1,1]-V[1,2]-V[2,1]+V[2,2]+(m[1]-m[2])^2
    mean = (log(A)+m_omega)/m_kappa
    println("upward messages")
    println(mean," ",A, " ", 2/(m_kappa^2*A^2))
    Message(GaussianMeanVariance, m=clamp(mean,tiny,huge),  v=clamp(2/(m_kappa^2*A^2),tiny,huge))

end

function ruleMGPCLinearGGDDD(   msg_out::Message{F1, Univariate},
                        msg_m::Message{F2, Univariate},
                        dist_v::ProbabilityDistribution{Univariate, F3},
                        dist_kappa::ProbabilityDistribution{Univariate},
                        dist_omega::ProbabilityDistribution{Univariate}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}

    d_out = convert(ProbabilityDistribution{Univariate,GaussianWeightedMeanPrecision},msg_out.dist)
    d_mean = convert(ProbabilityDistribution{Univariate,GaussianWeightedMeanPrecision},msg_m.dist)

    xi_mean = d_mean.params[:xi]
    w_mean = d_mean.params[:w]
    xi_out = d_out.params[:xi]
    w_out = d_out.params[:w]
    (m_v, v_v) =  unsafeMeanCov(dist_v)
    m_kappa = unsafeMean(dist_kappa)
    m_omega = unsafeMean(dist_omega)

    gamma = clamp(1/exp(m_kappa*m_v +(m_kappa^2)*v_v/2 +m_omega),tiny, huge)
    q_W = [w_out+gamma -gamma; -gamma w_mean+gamma]
    q_xi = [xi_out; xi_mean]
    println("marginals")
    println(q_W, " ", q_xi )


    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=q_xi, w=q_W)
end
