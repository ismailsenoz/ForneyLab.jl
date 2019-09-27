export
ruleSPGaussianMeanPrecisionOutVPP,
ruleSPGaussianMeanPrecisionMPVP,
ruleSPGaussianMeanPrecisionOutVGP,
ruleSPGaussianMeanPrecisionMGVP,
ruleVBGaussianMeanPrecisionM,
ruleVBGaussianMeanPrecisionW,
ruleVBGaussianMeanPrecisionOut,
ruleSVBGaussianMeanPrecisionOutVGD,
ruleSVBGaussianMeanPrecisionW,
ruleSVBGaussianMeanPrecisionMGVD,
ruleMGaussianMeanPrecisionGGD

ruleSPGaussianMeanPrecisionOutNPP(  msg_out::Nothing,
                                    msg_mean::Message{PointMass, V},
                                    msg_prec::Message{PointMass}) where V<:VariateType =
    Message(V, GaussianMeanPrecision, m=deepcopy(msg_mean.dist.params[:m]), w=deepcopy(msg_prec.dist.params[:m]))

ruleSPGaussianMeanPrecisionMPVP(msg_out::Message{PointMass}, msg_mean::Nothing, msg_prec::Message{PointMass}) =
    ruleSPGaussianMeanPrecisionOutVPP(msg_mean, msg_out, msg_prec)

function ruleSPGaussianMeanPrecisionOutNGP( msg_out::Nothing,
                                            msg_mean::Message{F, V},
                                            msg_prec::Message{PointMass}) where {F<:Gaussian, V<:VariateType}

    d_mean = convert(ProbabilityDistribution{V, GaussianMeanVariance}, msg_mean.dist)

    Message(V, GaussianMeanVariance, m=d_mean.params[:m], v=d_mean.params[:v] + cholinv(msg_prec.dist.params[:m]))
end

ruleSPGaussianMeanPrecisionMGVP(msg_out::Message{F}, msg_mean::Nothing, msg_prec::Message{PointMass}) where F<:Gaussian =
    ruleSPGaussianMeanPrecisionOutVGP(msg_mean, msg_out, msg_prec)

ruleVBGaussianMeanPrecisionM(   dist_out::ProbabilityDistribution{V},
                                dist_mean::Any,
                                dist_prec::ProbabilityDistribution) where V<:VariateType =
    Message(V, GaussianMeanPrecision, m=unsafeMean(dist_out), w=unsafeMean(dist_prec))

function ruleVBGaussianMeanPrecisionW(  dist_out::ProbabilityDistribution{Univariate},
                                        dist_mean::ProbabilityDistribution{Univariate},
                                        dist_prec::Any)

    (m_mean, v_mean) = unsafeMeanCov(dist_mean)
    (m_out, v_out) = unsafeMeanCov(dist_out)

    Message(Univariate, Gamma, a=1.5, b=0.5*(v_mean + v_out + (m_mean - m_out)^2))
end

function ruleVBGaussianMeanPrecisionW(  dist_out::ProbabilityDistribution{Multivariate},
                                        dist_mean::ProbabilityDistribution{Multivariate},
                                        dist_prec::Any)

    (m_mean, v_mean) = unsafeMeanCov(dist_mean)
    (m_out, v_out) = unsafeMeanCov(dist_out)

    Message(MatrixVariate, Wishart, v=cholinv( v_mean + v_out + (m_mean - m_out)*(m_mean - m_out)' ), nu=dims(dist_out) + 2.0)
end

ruleVBGaussianMeanPrecisionOut( dist_out::Any,
                                dist_mean::ProbabilityDistribution{V},
                                dist_prec::ProbabilityDistribution) where V<:VariateType =
    Message(V, GaussianMeanPrecision, m=unsafeMean(dist_mean), w=unsafeMean(dist_prec))

ruleSVBGaussianMeanPrecisionOutVGD(dist_out::Any,
                                   msg_mean::Message{F, V},
                                   dist_prec::ProbabilityDistribution) where{F<:Gaussian, V<:VariateType} =
    Message(V, GaussianMeanVariance, m=unsafeMean(msg_mean.dist), v=unsafeCov(msg_mean.dist) + cholinv(unsafeMean(dist_prec)))

function ruleSVBGaussianMeanPrecisionW(
    dist_out_mean::ProbabilityDistribution{Multivariate, F},
    dist_prec::Any) where F<:Gaussian

    joint_dims = dims(dist_out_mean)
    d_out_mean = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, dist_out_mean)
    (m, V) = unsafeMeanCov(d_out_mean)
    if joint_dims == 2
        return Message(Univariate, Gamma, a=1.5, b=0.5*(V[1,1] - V[1,2] - V[2,1] + V[2,2] + (m[1] - m[2])^2))
    else
        d = Int64(joint_dims/2)
        return Message(MatrixVariate, Wishart, v=cholinv( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end, 1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ), nu=d + 2.0)
    end
end

function ruleSVBGaussianMeanPrecisionMGVD(  msg_out::Message{F, V},
                                            dist_mean::Any,
                                            dist_prec::ProbabilityDistribution) where {F<:Gaussian, V<:VariateType}

    d_out = convert(ProbabilityDistribution{V, GaussianMeanVariance}, msg_out.dist)

    Message(V, GaussianMeanVariance, m=d_out.params[:m], v=d_out.params[:v] + cholinv(unsafeMean(dist_prec)))
end



function ruleMGaussianMeanPrecisionGGD(
    msg_out::Message{F1, V},
    msg_mean::Message{F2, V},
    dist_prec::ProbabilityDistribution) where {F1<:Gaussian, F2<:Gaussian, V<:VariateType}

    d_mean = convert(ProbabilityDistribution{V, GaussianWeightedMeanPrecision}, msg_mean.dist)
    d_out = convert(ProbabilityDistribution{V, GaussianWeightedMeanPrecision}, msg_out.dist)

    xi_y = d_out.params[:xi]
    W_y = d_out.params[:w]
    xi_m = d_mean.params[:xi]
    W_m = d_mean.params[:w]
    W_bar = unsafeMean(dist_prec)

    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_y; xi_m], w=[W_y+W_bar -W_bar; -W_bar W_m+W_bar])
end


function ruleSVBGaussianMeanPrecisionMGVD(  msg_out::Message{InverseLinearExponential},
                                            dist_mean::Any,
                                            dist_prec::ProbabilityDistribution)

    Psi = msg_out.dist.params[:psi]
    m_kappa = msg_out.dist.params[:k]
    m_omega = msg_out.dist.params[:w]


    expansion_point = (log(Psi)-m_omega)/m_kappa
    phi = exp(m_kappa*expansion_point+m_omega)
    gprime = -0.5*(m_kappa -  Psi*m_kappa/phi)
    gdprime = -0.5*(m_kappa^2*Psi/phi)

    # @show mean = (log(Psi)-m_omega)/m_kappa
    # var = cholinv(unsafeMean(dist_prec))+ 2/m_kappa^2
    var = -(1/gdprime)
    # @show g = -((m_kappa*expansion_point+m_omega)+Psi/phi)
    # @show newton_update = -g/gprime
    mean = expansion_point

    Message(Univariate,GaussianMeanVariance, m=mean, v=var+cholinv(unsafeMean(dist_prec)))
end

function ruleMGaussianMeanPrecisionGGD(
    msg_out::Message{InverseLinearExponential},
    msg_mean::Message{F1, V},
    dist_prec::ProbabilityDistribution) where {F1<:Gaussian, V<:VariateType}

    d_mean = convert(ProbabilityDistribution{V, GaussianWeightedMeanPrecision}, msg_mean.dist)
    Psi = msg_out.dist.params[:psi]
    m_kappa = msg_out.dist.params[:k]
    m_omega = msg_out.dist.params[:w]
    xi_m = d_mean.params[:xi]
    W_m = d_mean.params[:w]
    d_out = convert(ProbabilityDistribution{V, GaussianWeightedMeanPrecision}, ProbabilityDistribution(GaussianMeanVariance,m=(log(Psi)-m_omega)/m_kappa, v=2/m_kappa^2))
    xi_y = d_out.params[:xi]
    W_y = d_out.params[:w]
    W_bar = unsafeMean(dist_prec)

    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_y; xi_m], w=[W_y+W_bar -W_bar; -W_bar W_m+W_bar])
end

function ruleSVBGaussianMeanPrecisionMGVD(  msg_out::Message{LogLinearExponential},
                                            dist_mean::Any,
                                            dist_prec::ProbabilityDistribution)

    Psi = msg_out.dist.params[:psi]
    m_kappa = msg_out.dist.params[:k]
    m_omega = msg_out.dist.params[:w]


    expansion_point = (log(Psi)-m_omega)/m_kappa
    phi = exp(m_kappa*expansion_point+m_omega)
    gprime = -0.5*(m_kappa -  Psi*m_kappa/phi)
    gdprime = -0.5*(m_kappa^2*Psi/phi)

    # @show mean = (log(Psi)-m_omega)/m_kappa
    # var = cholinv(unsafeMean(dist_prec))+ 2/m_kappa^2
    var = -(1/gdprime)
    # @show g = -((m_kappa*expansion_point+m_omega)+Psi/phi)
    # @show newton_update = -g/gprime
    mean = expansion_point

    Message(Univariate,GaussianMeanVariance, m=mean, v=var+cholinv(unsafeMean(dist_prec)))
end

function ruleMGaussianMeanPrecisionGGD(
    msg_out::Message{LogLinearExponential},
    msg_mean::Message{F1, V},
    dist_prec::ProbabilityDistribution) where {F1<:Gaussian, V<:VariateType}

    d_mean = convert(ProbabilityDistribution{V, GaussianWeightedMeanPrecision}, msg_mean.dist)
    Psi = msg_out.dist.params[:psi]
    m_kappa = msg_out.dist.params[:k]
    m_omega = msg_out.dist.params[:w]
    xi_m = d_mean.params[:xi]
    W_m = d_mean.params[:w]
    d_out = convert(ProbabilityDistribution{V, GaussianWeightedMeanPrecision}, ProbabilityDistribution(GaussianMeanVariance,m=(log(Psi)-m_omega)/m_kappa, v=2/m_kappa^2))
    xi_y = d_out.params[:xi]
    W_y = d_out.params[:w]
    W_bar = unsafeMean(dist_prec)

    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_y; xi_m], w=[W_y+W_bar -W_bar; -W_bar W_m+W_bar])
end
