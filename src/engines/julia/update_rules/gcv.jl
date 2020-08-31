export
ruleSVBGCVGaussHermiteOutNGD,
ruleSVBGCVGaussHermiteMGND,
ruleSVBGCVGaussHermiteZDN,
ruleMGCVGaussHermiteMGGD,
ruleSVBGCVSphericalRadialOutNGD,
ruleSVBGCVSphericalRadialMGND,
ruleSVBGCVSphericalRadialZDN,
ruleMGCVSphericalRadialMGGD,
ruleSVBGCVLaplaceOutNGD,
ruleSVBGCVLaplaceMGND,
ruleSVBGCVLaplaceZDN,
ruleMGCVLaplaceMGGD,
ruleMGCVGaussHermiteFGD,
ruleSVBGCVGaussHermiteOutGFD,
ruleSVBGCVGaussHermiteMFGD,
prod!

function ruleSVBGCVGaussHermiteOutNGD(msg_out::Nothing,msg_m::Message{F, Multivariate},dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    ndims = dims(msg_m.dist)
    mean_m, cov_m = unsafeMeanCov(msg_m.dist)

    cubature = ghcubature(ndims, 20)

    Λ_out = approximate_kernel_expectation(cubature, (s) -> cholinv(g(s)), dist_z)

    return Message(Multivariate, GaussianMeanVariance,m=mean_m,v=cov_m+cholinv(Λ_out))
end

ruleSVBGCVGaussHermiteMGND(msg_out::Message{F, Multivariate},msg_m::Nothing, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian =
    ruleSVBGCVGaussHermiteOutNGD(msg_m, msg_out, dist_z, g)

## Many layers

function ruleSVBGCVGaussHermiteOutGFD(msg_out::Message{F, Multivariate},msg_m::Message{Function, Multivariate},dist_z::ProbabilityDistribution{Multivariate}, g::Function) where { F <: Gaussian }
    ndims = dims(msg_out.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    # mean_m, cov_m = unsafeMeanCov(msg_m.dist)

    cubature = ghcubature(ndims, 20)
    weights = getweights(cubature)
    points  = getpoints(cubature, mean_z, cov_z)

    mean_m, cov_v = approximate_meancov(cubature, msg_m.dist.params[:log_pdf], msg_out.dist)
    
    message = Message(Multivariate, GaussianMeanVariance, m = mean_m, v = cov_v)

    return ruleSVBGCVGaussHermiteOutNGD(nothing, message, dist_z, g)
end

ruleSVBGCVGaussHermiteMFGD(msg_out::Message{Function, Multivariate}, msg_m::Message{F, Multivariate}, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where { F <: Gaussian } = 
    ruleSVBGCVGaussHermiteOutGFD(msg_m, msg_out, dist_z, g)

## 

function ruleSVBGCVGaussHermiteZDN(dist_out_mean::ProbabilityDistribution{Multivariate}, msg_z::Nothing, g::Function)
    d = Int64(dims(dist_out_mean)/2)
    m_out_mean, cov_out_mean = unsafeMeanCov(dist_out_mean)
    psi = cov_out_mean[1:d,1:d] - cov_out_mean[1:d,d+1:end] - cov_out_mean[d+1:end, 1:d] + cov_out_mean[d+1:end,d+1:end] + (m_out_mean[1:d] - m_out_mean[d+1:end])*(m_out_mean[1:d] - m_out_mean[d+1:end])'

    l_pdf(z) = begin
        gz = g(z)
        -0.5*(logdet(gz) + tr(cholinv(gz)*psi))
    end

    return Message(Multivariate, Function, log_pdf = l_pdf, cubature = ghcubature(d, 20))
end


function ruleMGCVGaussHermiteMGGD(msg_out::Message{F1, Multivariate},msg_m::Message{F2, Multivariate},dist_z::ProbabilityDistribution{Multivariate,F3},g::Function) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
    ndims = dims(msg_out.dist)
    xi_out, Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    xi_mean, Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    dist_z = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z)

    cubature = ghcubature(ndims, 20)
    Λ = approximate_kernel_expectation(cubature, (z) -> cholinv(g(z)), dist_z)

    W = [ Λ + Λ_out -Λ; -Λ Λ_m + Λ] # + 1e-8*diageye(2*d)
    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_out;xi_mean], w = W)
end

function ruleMGCVGaussHermiteFGD(msg_out::Message{Function, Multivariate},msg_m::Message{F2, Multivariate},dist_z::ProbabilityDistribution{Multivariate,F3},g::Function) where { F2<:Gaussian,F3<:Gaussian }
    ndims = dims(msg_m.dist)
    # xi_out, Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    # xi_mean, Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    dist_z = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z)

    cubature = ghcubature(ndims, 20)
    Λ = approximate_kernel_expectation(cubature, (z) -> cholinv(g(z)), dist_z)

    # W = [ Λ + Λ_out -Λ; -Λ Λ_m + Λ] # + 1e-8*diageye(2*d)

    return ruleMGaussianMeanPrecisionFGD(msg_out, msg_m, ProbabilityDistribution(MatrixVariate, PointMass, m = Λ))
    # return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_out;xi_mean], w = W)
end

function ruleSVBGCVSphericalRadialOutNGD(msg_out::Nothing,msg_m::Message{F, Multivariate},dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    ndims = dims(msg_m.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    mean_m, cov_m = unsafeMeanCov(msg_m.dist)

    cubature = srcubature(ndims)

    # Λ_out = approximate_kernel_expectation(cubature, (s) -> cholinv(g(s)), dist_z)
    Λ_out = unscentedStatistics(cubature, (s) -> cholinv(g(s)), dist_z)

    return Message(Multivariate, GaussianMeanVariance,m=mean_m,v=cov_m+cholinv(Λ_out))
end

ruleSVBGCVSphericalRadialMGND(msg_out::Message{F, Multivariate},msg_m::Nothing, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian =
    ruleSVBGCVSphericalRadialOutNGD(msg_m,msg_out,dist_z,g)

function ruleSVBGCVSphericalRadialZDN(dist_out_mean::ProbabilityDistribution{Multivariate}, msg_z::Nothing, g::Function)
    d = Int64(dims(dist_out_mean)/2)
    m_out_mean, cov_out_mean = unsafeMeanCov(dist_out_mean)
    psi = cov_out_mean[1:d,1:d] - cov_out_mean[1:d,d+1:end] - cov_out_mean[d+1:end, 1:d] + cov_out_mean[d+1:end,d+1:end] + (m_out_mean[1:d] - m_out_mean[d+1:end])*(m_out_mean[1:d] - m_out_mean[d+1:end])'

    l_pdf(z) = begin
        gz = g(z)
        -0.5*(logdet(gz) + tr(cholinv(gz)*psi))
    end

    return Message(Multivariate, Function, log_pdf = l_pdf, cubature = srcubature(d))
end

function ruleMGCVSphericalRadialMGGD(msg_out::Message{F1, Multivariate},msg_m::Message{F2, Multivariate},dist_z::ProbabilityDistribution{Multivariate,F3},g::Function) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
    ndims = dims(msg_out.dist)
    xi_out, Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    xi_mean, Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)

    cubature = srcubature(ndims)

    # Λ = approximate_kernel_expectation(cubature, (s) -> cholinv(g(s)), dist_z)
    Λ = unscentedStatistics(cubature, (s) -> cholinv(g(s)), dist_z)

    W = [ Λ + Λ_out -Λ; -Λ Λ_m + Λ] # + 1e-8*diageye(2*d)
    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_out;xi_mean], w = W)

end

function ruleSVBGCVLaplaceOutNGD(msg_out::Nothing,msg_m::Message{F, Multivariate},dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    ndims = dims(msg_m.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    mean_m, cov_m = unsafeMeanCov(msg_m.dist)

    cubature = srcubature(ndims)

    # Λ = approximate_kernel_expectation(cubature, (s) -> cholinv(g(s)), dist_z)
    Λ = unscentedStatistics(cubature, (s) -> cholinv(g(s)), dist_z)

    return Message(Multivariate, GaussianMeanVariance,m=mean_m,v=cov_m+cholinv(Λ_out))
end

ruleSVBGCVLaplaceMGND(msg_out::Message{F, Multivariate},msg_m::Nothing, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian =
    ruleSVBGCVLaplaceOutNGD(msg_m, msg_out, dist_z, g)


function ruleSVBGCVLaplaceZDN(dist_out_mean::ProbabilityDistribution{Multivariate}, msg_z::Message{F, Multivariate}, g::Function) where F <: Gaussian
    d = Int64(dims(dist_out_mean)/2)
    m_out_mean, cov_out_mean = unsafeMeanCov(dist_out_mean)
    psi = cov_out_mean[1:d,1:d] - cov_out_mean[1:d,d+1:end] - cov_out_mean[d+1:end, 1:d] + cov_out_mean[d+1:end,d+1:end] + (m_out_mean[1:d] - m_out_mean[d+1:end])*(m_out_mean[1:d] - m_out_mean[d+1:end])'

    l_pdf(z) = begin
        gz = g(z)
        -0.5*(logdet(gz) + tr(cholinv(gz)*psi))
    end

    # epoint = ForneyLab.unsafeMean(msg_z.dist)
    # epoint[1] += tiny

    # mean, cov = NewtonMethod((s) -> l_pdf(s) + logPdf(msg_z.dist, s), epoint)

    # smoothRTSMessage(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)

    return Message(Multivariate, Function, log_pdf = l_pdf, cubature = nothing)
end

function ruleMGCVLaplaceMGGD(msg_out::Message{F1, Multivariate},msg_m::Message{F2, Multivariate},dist_z::ProbabilityDistribution{Multivariate,F3},g::Function) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
    ndims = dims(msg_out.dist)
    xi_out, Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    xi_mean, Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)

    cubature = srcubature(ndims)

    # Λ = approximate_kernel_expectation(cubature, (s) -> cholinv(g(s)), dist_z)
    Λ = unscentedStatistics(cubature, (s) -> cholinv(g(s)), dist_z)

    W = [ Λ + Λ_out -Λ; -Λ Λ_m + Λ] # + 1e-8*diageye(2*d)
    return ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_out;xi_mean], w = W)
end

@symmetrical function prod!(
    x::ProbabilityDistribution{Multivariate, Function},
    y::ProbabilityDistribution{Multivariate, F},
    z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(3), v=diageye(3))) where {F<:Gaussian}

    y = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},y)
    ndims = dims(y)

    # @show y

    g(s) = exp(x.params[:log_pdf](s))
    cubature = x.params[:cubature]

    m, V = approximate_meancov(cubature, g, y)

    z.params[:m] = m
    z.params[:v] = V
    return z
end

function approximate_meancov(::Nothing, g, distribution)
    epoint = ForneyLab.unsafeMean(distribution)
    epoint[1] += tiny
    mean, cov = NewtonMethod((s) -> log(g(s)) + logPdf(distribution, s), epoint)
    return mean, cov
end

import NLsolve: nlsolve, converged
import ReverseDiff

function NewtonMethod(g::Function, x_0::Array{Float64})
    dim = length(x_0)

    grad_g = (x) -> ForwardDiff.gradient(g, x)
    mode   = gradientOptimization(g, grad_g, x_0, 0.01)
    cov    = cholinv(-ForwardDiff.hessian(g, mode))

    return mode, cov
end

function NewtonMethod2(g::Function, x_0::Array{Float64})
    dim = length(x_0)

    grad = (s) -> -ReverseDiff.gradient(g, s)
    hess = (s) -> ForwardDiff.hessian(g, s)

    r = nlsolve(grad, hess, x_0, method = :newton, ftol = 1e-8)

    mode = r.zero

    @show r
    @show mode

    cov  = cholinv(-ForwardDiff.hessian(g, mode))

    return x, var
end

function NewtonMethod3(g::Function, x_0::Array{Float64})
    dim   = length(x_0)
    n_its = 25

    grad = (s) -> ReverseDiff.gradient(g, s)

    mode = x_0
    for i = 1:10
        grad = ReverseDiff.gradient(g, mode)
        hess = ForwardDiff.hessian(g, mode)
        var = -cholinv(hess)
        mode = mode + var * grad
    end

    @show mode

    # cov  = cholinv(-ForwardDiff.hessian(g, mode))

    return mode, nothing
end

function collectStructuredVariationalNodeInbounds(node::GCV, entry::ScheduleEntry)
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
            if entry.message_update_rule == SVBGCVLaplaceZDN
                # @show inbound_interface
                haskey(interface_to_schedule_entry, inbound_interface) || error("The GCV{Laplace} node's backward rule uses the incoming message on the input edge to determine the approximation point. Try altering the variable order in the scheduler to first perform a forward pass.")
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            elseif entry.message_update_rule == SVBGCVGaussHermiteOutGFD || entry.message_update_rule == SVBGCVGaussHermiteMFGD
                haskey(interface_to_schedule_entry, inbound_interface) || error("The GCV{Laplace} node's backward rule uses the incoming message on the input edge to determine the approximation point. Try altering the variable order in the scheduler to first perform a forward pass.")
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
                @show 1
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
