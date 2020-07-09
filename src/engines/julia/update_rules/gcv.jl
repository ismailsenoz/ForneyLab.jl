export
ruleSVBGCVCubatureOutNGD,
ruleSVBGCVCubatureMGND,
ruleSVBGCVCubatureZDN,
ruleMGCVCubatureMGGD,
ruleSVBGCVLaplaceOutNGD,
ruleSVBGCVLaplaceMGND,
ruleSVBGCVLaplaceZDN,
ruleMGCVLaplaceMGGD,
prod!

import LinearAlgebra: mul!, axpy!

function ruleSVBGCVCubatureOutNGD(msg_out::Nothing,msg_m::Message{F, Multivariate},dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    ndims = dims(msg_m.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    mean_m, cov_m = unsafeMeanCov(msg_m.dist)

    cubature = ghcubature(ndims, 20)
    weights = getweights(cubature)
    points  = getpoints(cubature, mean_z, cov_z)

    ginv = Base.Generator(points) do point
        return cholinv(g(point))
    end

    Λ_out = mapreduce(t -> t[1] * t[2], +, zip(weights, ginv)) / (pi ^ (ndims / 2))

    return Message(Multivariate, GaussianMeanVariance,m=mean_m,v=cov_m+cholinv(Λ_out))
end

function ruleSVBGCVCubatureMGND(msg_out::Message{F, Multivariate},msg_m::Nothing, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    ndims = dims(msg_out.dist)
    mean_out, cov_out = unsafeMeanCov(msg_out.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)

    cubature = ghcubature(ndims, 20)
    weights = getweights(cubature)
    points  = getpoints(cubature, mean_z, cov_z)

    ginv = Base.Generator(points) do point
        return cholinv(g(point))
    end

    Λ_m = mapreduce(t -> t[1] * t[2], +, zip(weights, ginv)) / (pi ^ (ndims / 2))

    return Message(Multivariate, GaussianMeanVariance,m=mean_out,v=cov_out+cholinv(Λ_m))
end

function ruleSVBGCVCubatureZDN(dist_out_mean::ProbabilityDistribution{Multivariate}, msg_z::Nothing, g::Function)
    d = Int64(dims(dist_out_mean)/2)
    m_out_mean, cov_out_mean = unsafeMeanCov(dist_out_mean)
    psi = cov_out_mean[1:d,1:d] - cov_out_mean[1:d,d+1:end] - cov_out_mean[d+1:end, 1:d] + cov_out_mean[d+1:end,d+1:end] + (m_out_mean[1:d] - m_out_mean[d+1:end])*(m_out_mean[1:d] - m_out_mean[d+1:end])'

    l_pdf(z) = begin
        gz = g(z)
        -0.5*(logdet(gz) + tr(cholinv(gz)*psi))
    end

    return Message(Multivariate, Function, log_pdf=l_pdf)
end


function ruleMGCVCubatureMGGD(msg_out::Message{F1, Multivariate},msg_m::Message{F2, Multivariate},dist_z::ProbabilityDistribution{Multivariate,F3},g::Function) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
    ndims = dims(msg_out.dist)
    xi_out, Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    xi_mean, Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    dist_z = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z)

    cubature = ghcubature(ndims, 20)
    Λ = approximate_kernel_expectation(cubature, (z) -> cholinv(g(z)), dist_z)

    W = [ Λ + Λ_out -Λ; -Λ Λ_m + Λ] # + 1e-8*diageye(2*d)
    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_out;xi_mean], w = W)

end

function collectStructuredVariationalNodeInbounds(node::GCV{Cubature}, entry::ScheduleEntry)
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]
    entry_posterior_factor = posteriorFactor(entry.interface.edge)
    local_posterior_factor_to_region = localPosteriorFactorToRegion(entry.interface.node)

    encountered_posterior_factors = Union{PosteriorFactor, Edge}[] # Keep track of encountered posterior factors
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_posterior_factor = posteriorFactor(node_interface.edge)

        if node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
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


    push!(inbounds, Dict{Symbol, Any}(:g => node.g,
                                      :keyword => false))

    return inbounds
end


function collectMarginalNodeInbounds(node::GCV, entry::MarginalEntry)
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry
    inbound_cluster = entry.target # Entry target is a cluster

    inbounds = Any[]
    entry_pf = posteriorFactor(first(entry.target.edges))
    encountered_external_regions = Set{Region}()
    for node_interface in entry.target.node.interfaces
        current_region = region(inbound_cluster.node, node_interface.edge) # Note: edges that are not assigned to a posterior factor are assumed mean-field
        current_pf = posteriorFactor(node_interface.edge) # Returns an Edge if no posterior factor is assigned
        inbound_interface = ultimatePartner(node_interface)

        if (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Edge is clamped, hard-code marginal of constant node
            push!(inbounds, assembleClamp!(copy(inbound_interface.node), ProbabilityDistribution)) # Copy Clamp before assembly to prevent overwriting dist_or_msg field
        elseif (current_pf === entry_pf)
            # Edge is internal, collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        elseif !(current_region in encountered_external_regions)
            # Edge is external and region is not yet encountered, collect marginal from marginal dictionary
            push!(inbounds, target_to_marginal_entry[current_region])
            push!(encountered_external_regions, current_region) # Register current region with encountered external regions
        end
    end

    push!(inbounds, Dict{Symbol, Any}(:g => node.g,
                                      :keyword => false))


    return inbounds
end

@symmetrical function prod!(
    x::ProbabilityDistribution{Multivariate, Function},
    y::ProbabilityDistribution{Multivariate, F},
    z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(3), v=diageye(3))) where {F<:Gaussian}

    y = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},y)
    ndims = dims(y)

    # @show y

    g(s) = exp(x.params[:log_pdf](s))

    cubature = ghcubature(ndims, 20)
    m, V = approximate_meancov(cubature, g, y)

    z.params[:m] = m
    z.params[:v] = V
    return z
end

import NLsolve: nlsolve

function NewtonMethod(g::Function,x_0::Array{Float64},n_its::Int64)
    dim = length(x_0)

    grad_g = (x) -> ForwardDiff.gradient(g, x)
    hess_g = (x) -> ForwardDiff.hessian(g, x)

    r = nlsolve(grad_g, hess_g, x_0, method = :newton, ftol = 1e-8)

    # @show x_0
    # @show r.zero
    # @show grad_g(r.zero)

    x = r.zero
    # @show -ForwardDiff.hessian(g, x)
    # @show r.trace
    cov = cholinv(-ForwardDiff.hessian(g, x))


    return x, cov
end
