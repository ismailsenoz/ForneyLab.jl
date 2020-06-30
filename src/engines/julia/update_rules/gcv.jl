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



function ruleSVBGCVCubatureOutNGD(msg_out::Nothing,msg_m::Message{F, Multivariate},dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    d = dims(msg_m.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    mean_m, cov_m = unsafeMeanCov(msg_m.dist)
    (sigma_points, weights_m, weights_c) = ForneyLab.sigmaPointsAndWeights(mean_z,cov_z)
    # Unscented approximation
    g_sigma = g.(sigma_points)
    Λ_out = sum([weights_m[k+1]*cholinv(g_sigma[k+1]) for k=0:2*d])

    return Message(Multivariate, GaussianMeanVariance,m=mean_m,v=cov_m+inv(Λ_out))
end

function ruleSVBGCVCubatureMGND(msg_out::Message{F, Multivariate},msg_m::Nothing, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    d = dims(msg_out.dist)
    mean_out, cov_out = unsafeMeanCov(msg_out.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    (sigma_points, weights_m, weights_c) = ForneyLab.sigmaPointsAndWeights(mean_z,cov_z)
    # Unscented approximation
    g_sigma = g.(sigma_points)
    Λ_m = sum([weights_m[k+1]*cholinv(g_sigma[k+1]) for k=0:2*d])

    return Message(Multivariate, GaussianMeanVariance,m=mean_out,v=cov_out+inv(Λ_m))
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
    d = dims(msg_out.dist)
    xi_out,Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    xi_mean,Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    dist_z = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z)
    h(z) = cholinv(g(z))
    Λ = kernelExpectation(h,dist_z,10)
    W = [Λ+Λ_out -Λ; -Λ Λ_m+Λ]+ 1e-8*diageye(2*d)
    return ProbabilityDistribution(Multivariate,GaussianWeightedMeanPrecision,xi=[xi_out;xi_mean],w=W)

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

const product = Iterators.product
const PIterator = Iterators.ProductIterator

using StaticArrays

const sigma_points = SA[-5.387480890011233, -4.603682449550744, -3.9447640401156265, -3.3478545673832163, -2.7888060584281296, -2.2549740020892757, -1.7385377121165857, -1.2340762153953255, -0.7374737285453978, -0.24534070830090382, 0.24534070830090382, 0.7374737285453978, 1.2340762153953255, 1.7385377121165857, 2.2549740020892757, 2.7888060584281296, 3.3478545673832163, 3.9447640401156265, 4.603682449550744, 5.387480890011233]
const sigma_weights = SA[2.2293936455341583e-13, 4.399340992273223e-10, 1.0860693707692783e-7, 7.802556478532184e-6, 0.00022833863601635774, 0.0032437733422378905, 0.024810520887463966, 0.10901720602002457, 0.28667550536283243, 0.4622436696006102, 0.4622436696006102, 0.28667550536283243, 0.10901720602002457, 0.024810520887463966, 0.0032437733422378905, 0.00022833863601635774, 7.802556478532184e-6, 1.0860693707692783e-7, 4.399340992273223e-10, 2.2293936455341583e-13]

function multiDimensionalPointsWeights(n::Int64,p::Int64)
    # sigma_points, sigma_weights = gausshermite(p)
    # points_iter = permmatrix(sigma_points, n)
    # weights_iter = permmatrix(sigma_weights, n)
    points_iter = product(repeat(SA[sigma_points], n)...)
    weights_iter = product(repeat(SA[sigma_weights], n)...)
    return points_iter, weights_iter
end

# using FastGaussQuadrature

import LinearAlgebra: mul!, adjoint!

function gaussHermiteCubature(g::Function,dist::ProbabilityDistribution{Multivariate,GaussianMeanVariance},p::Int64)
    d = dims(dist)
    m, P = ForneyLab.unsafeMeanCov(dist)
    sqrtP = sqrt(P)
    sqrt2 = sqrt(2)
    sqrtPi = sqrt(pi)
    points_iter, weights_iter = multiDimensionalPointsWeights(d,p)

    weights = prod.(weights_iter)
    tbuffer = similar(m, eltype(m), (d, ))
    pbuffer = similar(m, eltype(m), (d, ))
    points = Base.Generator(points_iter) do ptuple
        copyto!(pbuffer, ptuple)
        copyto!(tbuffer, m)
        mul!(tbuffer, sqrtP, pbuffer, sqrt2, 1.0)
    end

    cs = similar(m, eltype(m), length(points_iter))
    norm = 0.0
    mean = zeros(d)

    foreach(enumerate(zip(weights, points))) do (index, (weight, point))
        gv = g(point)
        cv = weight * gv

        broadcast!(*, point, point, cv)
        broadcast!(+, mean, mean, point)

        norm += cv

        @inbounds cs[index] = cv
    end

    # norm /= sqrtPi

    broadcast!(/, mean, mean, norm) # norm * sqrtPi

    cov  = zeros(d, d)
    tmp3 = similar(m, eltype(m), (d, d))
    foreach(enumerate(zip(points, cs))) do (index, (point, c))
        broadcast!(-, point, point, mean)
        mul!(tmp3, point, reshape(point, (1, d)))
        broadcast!(*, tmp3, tmp3, c)
        broadcast!(+, cov, cov, tmp3)
    end

    broadcast!(/, cov, cov, norm) # norm * sqrtPi)

    return mean, cov
end

@symmetrical function prod!(
    x::ProbabilityDistribution{Multivariate, Function},
    y::ProbabilityDistribution{Multivariate, F},
    z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(3), v=diageye(3))) where {F<:Gaussian}

    d_in1 = dims(y)
    y = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},y)
    m_y,cov_y = unsafeMeanCov(y)
    (sigma_points, weights_m, weights_c) = ForneyLab.sigmaPointsAndWeights(m_y,cov_y)
    g(s) = exp(x.params[:log_pdf](s))

    m, V = gaussHermiteCubature(g,y,20)

    z.params[:m] = m
    z.params[:v] = V
    return z
end

function NewtonMethod(g::Function,x_0::Array{Float64},n_its::Int64)
    dim = length(x_0)
    x = zeros(dim)
    var = zeros(dim,dim)
    for i=1:n_its
        grad = ForwardDiff.gradient(g,x_0)
        hessian = ForwardDiff.hessian(g,x_0)
        var = -cholinv(hessian)
        x = x_0 + var*grad
        x_0 = x
    end

    return x, var
end

function kernelExpectation(g::Function,dist::ProbabilityDistribution{Multivariate,GaussianMeanVariance},p::Int64)
    d = dims(dist)
    m, P = ForneyLab.unsafeMeanCov(dist)
    sqrtP = sqrt(P)
    sqrt2 = sqrt(2)
    sqrtPi = sqrt(pi)
    points_iter, weights_iter = multiDimensionalPointsWeights(d,p)
    #compute normalization constant

    weights = prod.(weights_iter)
    tbuffer = similar(m, eltype(m), (d, ))
    pbuffer = similar(m, eltype(m), (d, ))
    points = Base.Generator(points_iter) do ptuple
        copyto!(pbuffer, ptuple)
        copyto!(tbuffer, m)
        mul!(tbuffer, sqrtP, pbuffer, sqrt2, 1.0)
    end

    return mapreduce(r -> r[1] * g(r[2]), +, zip(weights, points), init = zeros(d, d))
    #
    # g_bar = zeros(d,d)
    # for (point_tuple, weights) in zip(points_iter, weights_iter)
    #     weight = prod(weights)
    #     point = collect(point_tuple)
    #     g_bar += weight.*g(m+sqrt(2).*sqrtP*point)
    # end
    # return g_bar./sqrt(pi)
end
