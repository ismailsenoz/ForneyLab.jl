export
ruleSPGaussianMeanVarianceOutNPP,
ruleSPGaussianMeanVarianceMPNP,
ruleSPGaussianMeanVarianceOutNGP,
ruleSPGaussianMeanVarianceMGNP,
ruleSPGaussianMeanVarianceVGGN,
ruleSPGaussianMeanVarianceVPGN,
ruleSPGaussianMeanVarianceOutNSP,
ruleVBGaussianMeanVarianceM,
ruleVBGaussianMeanVarianceOut

ruleSPGaussianMeanVarianceOutNPP(   msg_out::Nothing,
                                    msg_mean::Message{PointMass, V},
                                    msg_var::Message{PointMass}) where V<:VariateType =
    Message(V, GaussianMeanVariance, m=deepcopy(msg_mean.dist.params[:m]), v=deepcopy(msg_var.dist.params[:m]))

ruleSPGaussianMeanVarianceMPNP(msg_out::Message{PointMass}, msg_mean::Nothing, msg_var::Message{PointMass}) =
    ruleSPGaussianMeanVarianceOutNPP(msg_mean, msg_out, msg_var)

function ruleSPGaussianMeanVarianceOutNGP(  msg_out::Nothing,
                                            msg_mean::Message{F, V},
                                            msg_var::Message{PointMass}) where {F<:Gaussian, V<:VariateType}

    d_mean = convert(ProbabilityDistribution{V, GaussianMeanVariance}, msg_mean.dist)

    Message(V, GaussianMeanVariance, m=d_mean.params[:m], v=d_mean.params[:v] + msg_var.dist.params[:m])
end

ruleSPGaussianMeanVarianceMGNP(msg_out::Message{F}, msg_mean::Nothing, msg_var::Message{PointMass}) where F<:Gaussian =
    ruleSPGaussianMeanVarianceOutNGP(msg_mean, msg_out, msg_var)

function ruleSPGaussianMeanVarianceVGGN(msg_out::Message{F1, Univariate},
                                        msg_mean::Message{F2, Univariate},
                                        msg_var::Nothing) where {F1<:Gaussian, F2<:Gaussian}

    d_out  = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_out.dist)
    d_mean = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_mean.dist)

    Message(Univariate, Function, log_pdf=(x)-> -0.5*log(d_out.params[:v] + d_mean.params[:v] + x) - 1/(2*x)*(d_out.params[:m] - d_mean.params[:m])^2)
end

function ruleSPGaussianMeanVarianceVPGN(msg_out::Message{PointMass, Univariate},
                                        msg_mean::Message{F, Univariate},
                                        msg_var::Nothing) where F<:Gaussian

    d_mean = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_mean.dist)

    Message(Univariate, Function, log_pdf=(x)-> -0.5*log(d_mean.params[:v] + x) - 1/(2*x)*(msg_out.dist.params[:m] - d_mean.params[:m])^2)
end

#Bootstrap particle filter rule
function ruleSPGaussianMeanVarianceOutNSP(  msg_out::Nothing,
                                            msg_mean::Message{SampleList, V},
                                            msg_var::Message{PointMass}) where {V<:VariateType}

    samples = msg_mean.dist.params[:s]
    weights = msg_mean.dist.params[:w]
    n_samples = length(samples)
    new_samples = []
    for i=1:n_samples
        p = ProbabilityDistribution(V,GaussianMeanVariance,m=samples[i],v=msg_var.dist.params[:m])
        s = sample(p)
        push!(new_samples,s)
    end

    Message(V, SampleList, s=new_samples, w=weights)
end

ruleVBGaussianMeanVarianceM(dist_out::ProbabilityDistribution{V},
                            dist_mean::Any,
                            dist_var::ProbabilityDistribution) where V<:VariateType =
    Message(V, GaussianMeanVariance, m=unsafeMean(dist_out), v=unsafeMean(dist_var))

ruleVBGaussianMeanVarianceOut(  dist_out::Any,
                                dist_mean::ProbabilityDistribution{V},
                                dist_var::ProbabilityDistribution) where V<:VariateType =
    Message(V, GaussianMeanVariance, m=unsafeMean(dist_mean), v=unsafeMean(dist_var))
