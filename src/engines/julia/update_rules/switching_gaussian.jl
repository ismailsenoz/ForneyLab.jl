export
ruleSVBSwitchingGaussianOutNGDDD,
ruleSVBSwitchingGaussianMGNDDD,
ruleSVBSwitchingGaussianSDNDD,
ruleMSwitchingGaussianGGDDD


function ruleSVBSwitchingGaussianOutNGDDD(msg_out::Nothing,
                                          msg_m::Message{F,V},
                                          dist_s::ProbabilityDistribution,
                                          dist_A::ProbabilityDistribution,
                                          dist_Q::ProbabilityDistribution) where {F<:Gaussian, V<:VariateType}

    p = unsafeMean(dist_s) #category probabilities
    A = dist_A.params[:s] #SampleList
    Q = dist_Q.params[:s] #SampleList
    mean_m,cov_m = unsafeMeanCov(convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_m.dist))
    dim = length(mean_m)
    A_combination = zeros(dim,dim)
    Q_combination = zeros(dim,dim)
    for i=1:length(p)
        A_combination += p[i]*A[i]
        Q_combination += p[i]*Q[i]
    end
    return Message(V, GaussianMeanVariance,m=A_combination*mean_m,v=cov_m+Q_combination)

end
ruleSVBSwitchingGaussianMGNDDD(msg_out::Message{F,V},
                               msg_m::Nothing,
                               dist_s::ProbabilityDistribution,
                               dist_A::ProbabilityDistribution,
                               dist_Q::ProbabilityDistribution) where {F<:Gaussian, V<:VariateType} = ruleSVBSwitchingGaussianOutNGDDD(msg_m,msg_out,dist_s,dist_A,dist_Q)


function ruleSVBSwitchingGaussianSDNDD(dist_out_mean::ProbabilityDistribution,
                                       dist_s::Nothing,
                                       dist_A::ProbabilityDistribution,
                                       dist_Q::ProbabilityDistribution)
    A = dist_A.params[:s] #SampleList
    Q = dist_Q.params[:s] #SampleList
    m,Σ = unsafeMeanCov(convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, dist_out_mean))
    d = Int64(dims(dist_out_mean)/2)
    ψ = [(m[1:d]- A[i]*m[d+1:end])*(m[1:d]- A[i]*m[d+1:end])' + Σ[1:d,1:d]-A[i]*Σ[1:d,d+1:end] -
        Σ[d+1:end, 1:d]*A[i]'+ A[i]*Σ[d+1:end,d+1:end]*A[i]'  for i=1:length(dist_A.params[:s])]
    p = zeros(length(dist_A.params[:s]))
    for i=1:length(dist_A.params[:s])
        p[i] = exp(-0.5*(log(det(Q[i])) + tr(inv(Q[i])*ψ[i])))
    end
    return Message(Univariate,Categorical,p=p./sum(p))
end

function ruleMSwitchingGaussianGGDDD(msg_out::Message{F1,V},
                                     msg_m::Message{F2,V},
                                     dist_s::ProbabilityDistribution,
                                     dist_A::ProbabilityDistribution,
                                     dist_Q::ProbabilityDistribution) where {F1<:Gaussian,F2<:Gaussian, V<:VariateType}
    p = unsafeMean(dist_s) #category probabilities
    A = dist_A.params[:s] #SampleList
    Q = dist_Q.params[:s] #SampleList
    dim = dims(msg_m.dist)
    A_combination = zeros(dim,dim)
    Q_combination = zeros(dim,dim)
    for i=1:length(p)
        A_combination += p[i]*A[i]
        Q_combination += p[i]*Q[i]
    end
    fwd_message = Message(V,GaussianMeanVariance,m=A_combination*unsafeMean(msg_m.dist),v=A_combination*unsafeCov(msg_m.dist)*A_combination')
    return ruleMGaussianMeanPrecisionGGD(fwd_message,msg_out,ProbabilityDistribution(MatrixVariate,PointMass,m=Q_combination))
end
