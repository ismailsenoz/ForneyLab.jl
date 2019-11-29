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
ruleMGaussianMeanPrecisionGGD,
ruleSVBGaussianMeanPrecisionOutNED,
ruleSVBGaussianMeanPrecisionMEND,
ruleMGaussianMeanPrecisionGED,
ruleMGaussianMeanPrecisionEGD


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

function ruleSVBGaussianMeanPrecisionOutNED(msg_out::Message{F,Univariate},
                                   msg_mean::Message{ExponentialLinearQuadratic},
                                   dist_prec::ProbabilityDistribution) where F<:Gaussian
    dist_mean = msg_mean.dist
    message_prior = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_out,dist_prec)
    dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
    approx_dist = dist_prior*msg_mean.dist

    return Message(GaussianMeanVariance, m=unsafeMean(approx_dist), v=unsafeCov(approx_dist) + cholinv(unsafeMean(dist_prec)))
end

function ruleSVBGaussianMeanPrecisionMEND(msg_out::Message{ExponentialLinearQuadratic},
                                   msg_mean::Message{F, Univariate},
                                   dist_prec::ProbabilityDistribution) where F<:Gaussian

    dist_out = msg_out.dist
    message_prior = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_mean,dist_prec)
    dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
    approx_dist = dist_prior*msg_out.dist

    return Message(GaussianMeanVariance, m=unsafeMean(approx_dist), v=unsafeCov(approx_dist) + cholinv(unsafeMean(dist_prec)))
end


function ruleMGaussianMeanPrecisionGED(
    msg_out::Message{F, Univariate},
    msg_mean::Message{ExponentialLinearQuadratic},
    dist_prec::ProbabilityDistribution) where F<:Gaussian

    d_out = convert(ProbabilityDistribution{Univariate, GaussianWeightedMeanPrecision}, msg_out.dist)
    message_prior = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_mean,dist_prec)
    dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
    d_approx = dist_prior*msg_out.dist
    d_approx_mean = convert(ProbabilityDistribution{Univariate, GaussianWeightedMeanPrecision},d_approx)
    xi_y = d_out.params[:xi]
    W_y = d_out.params[:w]
    xi_m = d_approx_mean.params[:xi]
    W_m = d_approx_mean.params[:w]
    W_bar = unsafeMean(dist_prec)

    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_y; xi_m], w=[W_y+W_bar -W_bar; -W_bar W_m+W_bar])
end

function ruleMGaussianMeanPrecisionEGD(
    msg_out::Message{ExponentialLinearQuadratic},
    msg_mean::Message{F, Univariate},
    dist_prec::ProbabilityDistribution) where F<:Gaussian

    d_mean = convert(ProbabilityDistribution{Univariate, GaussianWeightedMeanPrecision}, msg_mean.dist)
    message_prior = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_mean,dist_prec)
    dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
    d_approx = dist_prior*msg_out.dist
    d_approx_out = convert(ProbabilityDistribution{Univariate, GaussianWeightedMeanPrecision},d_approx)
    xi_y = d_approx_out.params[:xi]
    W_y = d_approx_out.params[:w]
    xi_m = d_mean.params[:xi]
    W_m = d_mean.params[:w]
    W_bar = unsafeMean(dist_prec)

    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_y; xi_m], w=[W_y+W_bar -W_bar; -W_bar W_m+W_bar])
end

# ###Custom inbounds
function collectStructuredVariationalNodeInbounds(node::GaussianMeanPrecision, entry::ScheduleEntry, interface_to_msg_idx::Dict{Interface, Int})
    # Collect inbounds
    inbounds = String[]
    entry_recognition_factor_id = recognitionFactorId(entry.interface.edge)
    local_cluster_ids = localRecognitionFactorization(entry.interface.node)

    recognition_factor_ids = Symbol[] # Keep track of encountered recognition factor ids
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        node_interface_recognition_factor_id = recognitionFactorId(node_interface.edge)

        if node_interface == entry.interface
            # Ignore marginal of outbound edge
            if (entry.msg_update_rule == SVBGaussianMeanPrecisionOutNED) || (entry.msg_update_rule == SVBGaussianMeanPrecisionMEND)
                inbound_idx = interface_to_msg_idx[inbound_interface]
                push!(inbounds, "messages[$inbound_idx]")
            else
                push!(inbounds, "nothing")
            end
        elseif (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, marginalString(inbound_interface.node))
        elseif node_interface_recognition_factor_id == entry_recognition_factor_id
            # Collect message from previous result
            inbound_idx = interface_to_msg_idx[inbound_interface]
            push!(inbounds, "messages[$inbound_idx]")
        elseif !(node_interface_recognition_factor_id in recognition_factor_ids)
            # Collect marginal from marginal dictionary (if marginal is not already accepted)
            marginal_idx = local_cluster_ids[node_interface_recognition_factor_id]
            push!(inbounds, "marginals[:$marginal_idx]")
        end

        push!(recognition_factor_ids, node_interface_recognition_factor_id)
    end

    return inbounds
end
