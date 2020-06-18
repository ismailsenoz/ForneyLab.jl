export
ruleSVBSwitchingGaussianOutNGD,
ruleSVBSwitchingGaussianMGND,
ruleSVBSwitchingGaussianSDN,
ruleMSwitchingGaussianGGD


function ruleSVBSwitchingGaussianOutNGD(msg_out::Nothing,
                                          msg_m::Message{F,V},
                                          dist_s::ProbabilityDistribution, A::Vector{Matrix{Float64}},Q::Vector{Matrix{Float64}}) where {F<:Gaussian, V<:VariateType}

    p = unsafeMean(dist_s) #category probabilities
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
ruleSVBSwitchingGaussianMGND(msg_out::Message{F,V},
                               msg_m::Nothing,
                               dist_s::ProbabilityDistribution, A::Vector{Matrix{Float64}},Q::Vector{Matrix{Float64}}) where {F<:Gaussian, V<:VariateType} = ruleSVBSwitchingGaussianOutNGD(msg_m,msg_out,dist_s,A,Q)


function ruleSVBSwitchingGaussianSDN(dist_out_mean::ProbabilityDistribution,
                                       dist_s::Nothing, A::Vector{Matrix{Float64}},Q::Vector{Matrix{Float64}})
    m,Σ = unsafeMeanCov(convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, dist_out_mean))
    d = Int64(dims(dist_out_mean)/2)
    ψ = [(m[1:d]- A[i]*m[d+1:end])*(m[1:d]- A[i]*m[d+1:end])' + Σ[1:d,1:d]-A[i]*Σ[1:d,d+1:end] -
        Σ[d+1:end, 1:d]*A[i]'+ A[i]*Σ[d+1:end,d+1:end]*A[i]'  for i=1:size(A[1])[1]]
    p = zeros(size(A[1])[1])
    for i=1:size(A[1])[1]
        p[i] = exp(-0.5*(log(det(Q[i])) + tr(inv(Q[i])*ψ[i])))
    end
    return Message(Univariate,Categorical,p=p./sum(p))
end

function ruleMSwitchingGaussianGGD(msg_out::Message{F1,V},
                                     msg_m::Message{F2,V},
                                     dist_s::ProbabilityDistribution,A::Vector{Matrix{Float64}},Q::Vector{Matrix{Float64}}) where {F1<:Gaussian,F2<:Gaussian, V<:VariateType}
    p = unsafeMean(dist_s) #category probabilities
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

function collectStructuredVariationalNodeInbounds(node::SwitchingGaussian, entry::ScheduleEntry)
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

    push!(inbounds, Dict{Symbol, Any}(:A => node.A,
                                      :Q => node.Q,
                                      :keyword => true))

    return inbounds
end

function collectMarginalNodeInbounds(node::SwitchingGaussian, entry::MarginalEntry)
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
    push!(inbounds, Dict{Symbol, Any}(:A => node.A,
                                      :Q => node.Q,
                                      :keyword => true))
    return inbounds
end
