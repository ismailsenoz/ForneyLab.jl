export
ruleSVBMultivariateGPCOutNGD,
ruleSVBMultivariateGPCMeanGND,
ruleSVBMultivariateGPCCovDN,
ruleMMultivariateGPCGGD

import Distributions.MvNormal
using LinearAlgebra

# function expectation_dummy(g::Function,m::Array, A::Matrix)
#     return g(m)+A
# end

function G(g::Function,A::Matrix, z::Array)
    return inv(g(z)+A)
end

function expectation(q::ProbabilityDistribution, g::Function, A::Matrix)
    alpha = 1.0
    kappa = 1.0
    (mq, vq) = ForneyLab.unsafeMeanCov(q)
    dim = length(mq)
    sigma_points = sigmaPoints(mq, vq, alpha,kappa)
    weights = weights_mean(alpha,kappa,dim)
    points_non_linear = Array{Float64,3}(undef,2*dim+1,dim,dim)
    mean = zeros(dim,dim)
    for i = 1:2*dim+1
        points_non_linear[i,:,:] = G(g,A,sigma_points[i,:])
        mean += weights[i]*points_non_linear[i,:,:]
    end

    return mean
end

function weights_mean(alpha::Float64,kappa::Float64, dim::Int64)
    weights = Array{Float64}(undef,2*dim+1)
    lambda = alpha^2*(dim+kappa) - dim
    weights[1] = lambda/(dim+lambda)
    for i = 2:2*dim+1
        weights[i] = 1/(2*(dim+lambda))
    end
    return weights
end


function sigmaPoints(m::Array, P::Matrix, alpha::Float64,kappa::Float64)
    dim = length(m)
    sigma_points = Array{Float64,2}(undef,2*dim+1,dim)
    sigma_points[1,:] = m
    lambda = alpha^2*(dim+kappa) - dim
    chol_P = sqrt(Hermitian(P))
    scale = sqrt(dim+lambda)
    for i = 1:dim
        sigma_points[i+1,:] = m + scale*chol_P[:,i]
        sigma_points[dim+1+i,:] = m - scale*chol_P[:,i]
    end
    return sigma_points
end

function ruleSVBMultivariateGPCOutNGD(dist_out::Nothing,
                                      msg_mean::Message{F, Multivariate},
                                      dist_cov::ProbabilityDistribution,
                                      g::Function) where F<:Gaussian

    d_mean = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_mean.dist)
    phi = inv(expectation(dist_cov, g, d_mean.params[:v]))
    # println("out rule")
    # isposdef(phi)
    return Message(Multivariate, GaussianMeanVariance, m=d_mean.params[:m], v=phi)
end

function ruleSVBMultivariateGPCMeanGND(msg_out::Message{F, Multivariate},
                                       dist_mean::Nothing,
                                       dist_cov::ProbabilityDistribution,
                                       g::Function) where F<:Gaussian

    d_out = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_out.dist)
    phi = inv(expectation(dist_cov, g, d_out.params[:v]))
    # println("mean rule")
    # isposdef(phi)
    return Message(Multivariate, GaussianMeanVariance, m=d_out.params[:m], v=phi)
end

function ruleSVBMultivariateGPCCovDN(dist_out_mean::ProbabilityDistribution,
                                     dist_cov::Nothing,
                                     g::Function)

    (out_mean_m, out_mean_cov) = ForneyLab.unsafeMeanCov(dist_out_mean)
    joint_dims = dims(dist_out_mean)
    d = Int64(joint_dims/2)
    Psi = out_mean_cov[1:d,1:d] + out_mean_cov[d+1:end,d+1:end] + (out_mean_m[1:d] - out_mean_m[d+1:end])*(out_mean_m[1:d] - out_mean_m[d+1:end])'
    # println("cov rule")
    # println(isposdef(Psi))
    return Message(LogDetTrace, g=g, Psi=Psi, cov=out_mean_cov[d+1:end,d+1:end])
end

function ruleMMultivariateGPCGGD(msg_out::Message{F1, Multivariate},
                                 msg_mean::Message{F2, Multivariate},
                                 dist_cov::ProbabilityDistribution,
                                 g::Function) where {F1<:Gaussian, F2<:Gaussian}


    d_out = convert(ProbabilityDistribution{Multivariate, GaussianWeightedMeanPrecision}, msg_out.dist)
    d_mean = convert(ProbabilityDistribution{Multivariate, GaussianWeightedMeanPrecision}, msg_mean.dist)
    z = zeros(dims(d_out),dims(d_out))
    phi = expectation(dist_cov,g,z)
    lambda = Hermitian([(phi+d_out.params[:w])  -phi; -phi  (phi+d_mean.params[:w])])
    weighted_mean = [d_out.params[:xi]; d_mean.params[:xi]]

    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=weighted_mean, w=lambda)
end

function collectStructuredVariationalNodeInbounds(node::MultivariateGPC, entry::ScheduleEntry, interface_to_msg_idx::Dict{Interface, Int})
    # Collect inbounds
    inbounds = String[]
    entry_recognition_factor_id = recognitionFactorId(entry.interface.edge)
    local_cluster_ids = localRecognitionFactorization(entry.interface.node)

    recognition_factor_ids = Symbol[] # Keep track of encountered recognition factor ids
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        partner_node = inbound_interface.node
        node_interface_recognition_factor_id = recognitionFactorId(node_interface.edge)

        if node_interface == entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, "nothing")
        elseif isa(partner_node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, marginalString(partner_node))
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

    push!(inbounds, "$(node.g)")
    return inbounds
end


function collectMarginalNodeInbounds(node::MultivariateGPC, entry::MarginalScheduleEntry, interface_to_msg_idx::Dict{Interface, Int})
    # Collect inbounds
    inbounds = String[]
    entry_recognition_factor_id = recognitionFactorId(first(entry.target.edges))
    local_cluster_ids = localRecognitionFactorization(entry.target.node)

    recognition_factor_ids = Symbol[] # Keep track of encountered recognition factor ids
    for node_interface in entry.target.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        partner_node = inbound_interface.node
        node_interface_recognition_factor_id = recognitionFactorId(node_interface.edge)

        if isa(partner_node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, marginalString(partner_node))
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

    push!(inbounds, "$(node.g)")

    return inbounds
end
