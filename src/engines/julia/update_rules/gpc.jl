export
ruleVBGPCOutVPP,
ruleSVBGPCOutVGG,
ruleSVBGPCMeanGVG,
ruleSVBGPCVarGV,
ruleMGPCGGD


function ruleVBGPCOutVPP(   msg_out::Nothing,
                            dist_mean::ProbabilityDistribution{Univariate, PointMass},
                            dist_v::ProbabilityDistribution{Univariate, PointMass})
    (m_mean) = dist_mean.params[:m]
    (m_v) = dist_v.params[:m]

    gamma = exp(m_v)

    Message(GaussianMeanVariance, m=m_mean,  v=gamma)
end

function ruleSVBGPCOutVGG(  msg_out::Nothing,
                            msg_mean::Message{F,Univariate},
                            dist_v::ProbabilityDistribution{Univariate}) where F<:Gaussian

    d_mean = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance}, msg_mean.dist)

    m_v =  unsafeMean(dist_v)
    v_v = unsafeCov(dist_v)
    m_mean = d_mean.params[:m]
    v_mean = d_mean.params[:v]
    gamma = exp(m_v - v_v/2)

    # println(m_mean," ", v_mean+gamma)
    Message(GaussianMeanVariance, m=clamp(m_mean,tiny,huge),  v=clamp(v_mean + gamma,tiny,huge))
end

function ruleSVBGPCMeanGVG( msg_out::Message{F,Univariate},
                            msg_mean::Nothing,
                            dist_v::ProbabilityDistribution) where F<:Union{Gaussian, PointMass}

    d_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)

    m_out = d_out.params[:m]
    v_out = d_out.params[:v]
    m_v =  unsafeMean(dist_v)
    v_v = unsafeCov(dist_v)
    gamma = exp(m_v - v_v/2)

    # println(m_out," ", v_out+gamma)
    Message(GaussianMeanVariance, m=clamp(m_out,tiny,huge),  v=clamp(v_out + gamma,tiny,huge))
end

function ruleSVBGPCVarGV(   dist_out_mean::ProbabilityDistribution{Multivariate},
                            dist_v::Nothing)

    (m, V) = unsafeMeanCov(dist_out_mean)

    A = V[1,1]-V[1,2]-V[2,1]+V[2,2]+(m[1]-m[2])^2

    # println(log(A)," ", 2/A^2)
    Message(GaussianMeanVariance, m=clamp(log(A),tiny,huge),  v=clamp(2/(A^2),tiny,huge))

end

function ruleMGPCGGD(   msg_out::Message{F1, Univariate},
                        msg_m::Message{F2, Univariate},
                        dist_v::ProbabilityDistribution{Univariate, F3}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}

    d_out = convert(ProbabilityDistribution{Univariate,GaussianWeightedMeanPrecision},msg_out.dist)
    d_mean = convert(ProbabilityDistribution{Univariate,GaussianWeightedMeanPrecision},msg_m.dist)

    xi_mean = d_mean.params[:xi]
    w_mean = d_mean.params[:w]
    xi_out = d_out.params[:xi]
    w_out = d_out.params[:w]
    (m_v, v_v) =  unsafeMeanCov(dist_v)

    gamma = clamp(exp(-m_v +v_v/2 ),tiny, huge)
    q_W = [w_out+gamma -gamma; -gamma w_mean+gamma]
    q_xi = [xi_out; xi_mean]
    # println("marginals")
    # println(q_W, " ", q_xi )
    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=q_xi, w=q_W)


end
