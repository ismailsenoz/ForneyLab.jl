export
ruleSPGaussianMeanPrecisionOutNPP,
ruleSPGaussianMeanPrecisionMPNP,
ruleSPGaussianMeanPrecisionOutNGP,
ruleSPGaussianMeanPrecisionMGNP,
ruleVBGaussianMeanPrecisionM,
ruleVBGaussianMeanPrecisionW,
ruleVBGaussianMeanPrecisionOut,
ruleSVBGaussianMeanPrecisionOutVGD,
ruleSVBGaussianMeanPrecisionW,
ruleSVBGaussianMeanPrecisionMGVD,
ruleSVBGaussianMeanPrecisionMFND,
ruleMGaussianMeanPrecisionFGD,
ruleMGaussianMeanPrecisionGGD

ruleSPGaussianMeanPrecisionOutNPP(  msg_out::Nothing,
                                    msg_mean::Message{PointMass, V},
                                    msg_prec::Message{PointMass}) where V<:VariateType =
    Message(V, GaussianMeanPrecision, m=deepcopy(msg_mean.dist.params[:m]), w=deepcopy(msg_prec.dist.params[:m]))

ruleSPGaussianMeanPrecisionMPNP(msg_out::Message{PointMass}, msg_mean::Nothing, msg_prec::Message{PointMass}) =
    ruleSPGaussianMeanPrecisionOutNPP(msg_mean, msg_out, msg_prec)

function ruleSPGaussianMeanPrecisionOutNGP( msg_out::Nothing,
                                            msg_mean::Message{F, V},
                                            msg_prec::Message{PointMass}) where {F<:Gaussian, V<:VariateType}

    d_mean = convert(ProbabilityDistribution{V, GaussianMeanVariance}, msg_mean.dist)

    Message(V, GaussianMeanVariance, m=d_mean.params[:m], v=d_mean.params[:v] + cholinv(msg_prec.dist.params[:m]))
end

ruleSPGaussianMeanPrecisionMGNP(msg_out::Message{F}, msg_mean::Nothing, msg_prec::Message{PointMass}) where F<:Gaussian =
    ruleSPGaussianMeanPrecisionOutNGP(msg_mean, msg_out, msg_prec)

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

function ruleSVBGaussianMeanPrecisionMFND(msg_out::Message{Function,Multivariate},
                                          msg_mean::Message{F,Multivariate},
                                          dist_prec::ProbabilityDistribution) where F<:Gaussian

    msg_fwd = ruleSVBGaussianMeanPrecisionOutVGD(nothing,msg_mean,dist_prec)
    m_mean,v_mean = unsafeMeanCov(msg_fwd.dist*msg_out.dist)

    return Message(Multivariate,GaussianMeanVariance,m=m_mean,v=v_mean+inv(unsafeMean(dist_prec)))
end

function ruleMGaussianMeanPrecisionFGD(msg_out::Message{Function,Multivariate},
                                       msg_mean::Message{F,Multivariate},
                                       dist_prec::ProbabilityDistribution) where F<:Gaussian
    d = dims(msg_mean.dist)
    m_mean,v_mean = unsafeMeanCov(msg_mean.dist)
    Wbar = unsafeMean(dist_prec)
    W = [Wbar -Wbar; -Wbar Wbar]
    # f(s) =  exp.(msg_out.dist.params[:log_pdf](s))
    # h(s) = exp.(-0.5.* (s .- m_mean)'*cholinv(v_mean)*(s .- m_mean))
    l(z) = @views exp.(-0.5 * z'*W*z - 0.5 * (z[d+1:end] - m_mean)' * cholinv(v_mean) * (z[d+1:end] - m_mean) + msg_out.dist.params[:log_pdf](z[1:d]))
    #Expansion point
    msg_fwd = ruleSVBGaussianMeanPrecisionOutVGD(nothing,msg_mean,dist_prec)
    point1 = unsafeMean(msg_fwd.dist*msg_out.dist)

    m_joint, v_joint = NewtonMethod(l,[point1;m_mean],10)
    return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=m_joint,v=v_joint)
end

function collectStructuredVariationalNodeInbounds(node::GaussianMeanPrecision, entry::ScheduleEntry)
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]
    entry_posterior_factor = posteriorFactor(entry.interface.edge)
    local_posterior_factor_to_region = localPosteriorFactorToRegion(entry.interface.node)

    encountered_posterior_factors = Union{PosteriorFactor, Edge}[] # Keep track of encountered posterior factors# Keep track of encountered posterior factors
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_posterior_factor = posteriorFactor(node_interface.edge)

        if node_interface === entry.interface
            if (entry.message_update_rule == SVBGaussianMeanPrecisionMFND)
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            else
                push!(inbounds, nothing)
            end

        elseif (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        elseif current_posterior_factor === entry_posterior_factor
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        elseif !(current_posterior_factor in encountered_posterior_factors)
            # Collect marginal from marginal dictionary (if marginal is not already accepted)
            target = local_posterior_factor_to_region[current_posterior_factor]
            push!(inbounds, target_to_marginal_entry[target])
        end

        push!(encountered_posterior_factors, current_posterior_factor)
    end

    return inbounds
end
