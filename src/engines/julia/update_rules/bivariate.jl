export
ruleSPBivariateLIn1MNG,
ruleSPBivariateLIn2MGN,
ruleSPBivariateLOutNGG,
ruleMBivariateLOutNGG

function ruleSPBivariateLOutNGG(msg_out::Nothing, msg_in1::Message{F1, Univariate}, msg_in2::Message{F2, Univariate}, g::Function, status::Dict, n_samples::Int) where {F1<:Gaussian,F2<:Gaussian}
    # The forward message is parameterized by a SampleList
    dist_in1 = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_in1.dist)
    dist_in2 = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_in2.dist)

    samples1 = dist_in1.params[:m] .+ sqrt(dist_in1.params[:v]).*randn(n_samples)
    samples2 = dist_in2.params[:m] .+ sqrt(dist_in2.params[:v]).*randn(n_samples)

    sample_list = g.(samples1, samples2)
    weight_list = ones(n_samples)/n_samples

    if length(sample_list[1]) == 1
        return Message(Univariate, SampleList, s=sample_list, w=weight_list)
    else
        return Message(Multivariate, SampleList, s=sample_list, w=weight_list)
    end
end

function ruleSPBivariateLOutNGG(msg_out::Nothing, msg_in1::Message{F1, Multivariate}, msg_in2::Message{F2, Univariate}, g::Function, status::Dict, n_samples::Int) where {F1<:Gaussian,F2<:Gaussian}
    # The forward message is parameterized by a SampleList
    dist_in1 = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_in1.dist)
    dist_in2 = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_in2.dist)

    C1L = cholesky(dist_in1.params[:v]).L
    dim = dims(dist_in1)
    samples1 = Vector{Vector{Float64}}(undef, n_samples)
    
    for j=1:n_samples
        samples1[j] = dist_in1.params[:m] + C1L*randn(dim)
    end
    samples2 = dist_in2.params[:m] .+ sqrt(dist_in2.params[:v]).*randn(n_samples)

    sample_list = g.(samples1, samples2)
    weight_list = ones(n_samples)/n_samples

    if length(sample_list[1]) == 1
        return Message(Univariate, SampleList, s=sample_list, w=weight_list)
    else
        return Message(Multivariate, SampleList, s=sample_list, w=weight_list)
    end
end

function ruleSPBivariateLOutNGG(msg_out::Nothing, msg_in1::Message{F1, Univariate}, msg_in2::Message{F2, Multivariate}, g::Function, status::Dict, n_samples::Int) where {F1<:Gaussian,F2<:Gaussian}
    # The forward message is parameterized by a SampleList
    dist_in1 = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, msg_in1.dist)
    dist_in2 = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_in2.dist)

    samples1 = dist_in1.params[:m] .+ sqrt(dist_in1.params[:v]).*randn(n_samples)

    C2L = cholesky(dist_in2.params[:v]).L
    dim = dims(dist_in2)
    samples2 = Vector{Vector{Float64}}(undef, n_samples)
    
    for j=1:n_samples
        samples2[j] = dist_in2.params[:m] + C2L*randn(dim)
    end

    sample_list = g.(samples1, samples2)
    weight_list = ones(n_samples)/n_samples

    if length(sample_list[1]) == 1
        return Message(Univariate, SampleList, s=sample_list, w=weight_list)
    else
        return Message(Multivariate, SampleList, s=sample_list, w=weight_list)
    end
end

function ruleSPBivariateLOutNGG(msg_out::Nothing, msg_in1::Message{F1, Multivariate}, msg_in2::Message{F2, Multivariate}, g::Function, status::Dict, n_samples::Int) where {F1<:Gaussian,F2<:Gaussian}
    # The forward message is parameterized by a SampleList
    dist_in1 = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_in1.dist)
    dist_in2 = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, msg_in2.dist)

    C1L = cholesky(dist_in1.params[:v]).L
    dim1 = dims(dist_in1)
    samples1 = Vector{Vector{Float64}}(undef, n_samples)

    C2L = cholesky(dist_in2.params[:v]).L
    dim2 = dims(dist_in2)
    samples2 = Vector{Vector{Float64}}(undef, n_samples)

    for j=1:n_samples
        samples1[j] = dist_in1.params[:m] + C1L*randn(dim1)
        samples2[j] = dist_in2.params[:m] + C2L*randn(dim2)
    end

    sample_list = g.(samples1, samples2)
    weight_list = ones(n_samples)/n_samples

    if length(sample_list[1]) == 1
        return Message(Univariate, SampleList, s=sample_list, w=weight_list)
    else
        return Message(Multivariate, SampleList, s=sample_list, w=weight_list)
    end
end

function approxMessageBivariate(m_prior::Number,v_prior::Number,m_post::Number,v_post::Number)

    if abs(v_prior-v_post) < 1e-5
        v_message = 1e-5
    else
        v_message = v_prior*v_post/(v_prior-v_post)
    end
    m_message = (m_post*(v_prior+v_message) - m_prior*v_message)/v_prior
    return Message(Univariate, GaussianMeanVariance, m=m_message, v=v_message)
end

function approxMessageBivariate(m_prior::Array,v_prior,m_post::Array,v_post)

    w_prior, w_post = inv(v_prior+2e-5*diageye(length(m_prior))), inv(v_post+1e-5*diageye(length(m_prior)))
    w_message = w_post - w_prior
    xi_message = (w_prior+w_message)*m_post - w_prior*m_prior
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi_message, w=w_message)
end

function ruleSPBivariateLIn1MNG(msg_out::Message{Fout, Vout}, msg_in1::Message{F1, V1}, msg_in2::Message{F2, V2}, g::Function, status::Dict, n_samples::Int) where {Fout<:SoftFactor, Vout<:VariateType, F1<:Gaussian, V1<:VariateType, F2<:Gaussian, V2<:VariateType}

    if status[:updated]
        status[:updated] = false
        return status[:message]
    else
        dist_in1 = convert(ProbabilityDistribution{V1, GaussianMeanVariance}, msg_in1.dist)
        dist_in2 = convert(ProbabilityDistribution{V2, GaussianMeanVariance}, msg_in2.dist)

        m_concat, v_concat, dim1, dim2 = mergeInputs(dist_in1, dist_in2)

        dim_tot = dim1 + dim2
        
        log_prior_pdf(x) = -0.5*(dim_tot*log(2pi) + log(det(v_concat)) + transpose(x-m_concat)*inv(v_concat)*(x-m_concat))

        function log_joint_dims(s::Array,dim1::Int64,dim2::Int64)
            if dim1 == 1
                if dim2 == 1
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1]::Number,s[end]::Number))
                else
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1],s[2:end]))
                end
            else
                if dim2 == 1
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1:dim1],s[end]))
                else
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1:dim1],s[dim1+1:end]))
                end
            end
        end

        log_joint(s) = log_joint_dims(s,dim1,dim2)
        d_log_joint(s) = ForwardDiff.gradient(log_joint, s)
        
        m_post = gradientOptimization(log_joint, d_log_joint, m_concat, 0.01)
        var_post = Hermitian(inv(- 1.0 .* ForwardDiff.jacobian(d_log_joint, m_post)))

        status[:updated] = true

        mean1, var1, mean2, var2 = decomposePosteriorParameters(dist_in1, dist_in2, m_post, var_post)
        
        status[:message] = approxMessageBivariate(dist_in2.params[:m],dist_in2.params[:v], mean2, var2)
        return approxMessageBivariate(dist_in1.params[:m],dist_in1.params[:v],mean1,var1)
    end

end

function ruleSPBivariateLIn2MGN(msg_out::Message{Fout, Vout}, msg_in1::Message{F1, V1}, msg_in2::Message{F2, V2}, g::Function, status::Dict, n_samples::Int) where {Fout<:SoftFactor, Vout<:VariateType, F1<:Gaussian, V1<:VariateType, F2<:Gaussian, V2<:VariateType}

    if status[:updated]
        status[:updated] = false
        return status[:message]
    else
        dist_in1 = convert(ProbabilityDistribution{V1, GaussianMeanVariance}, msg_in1.dist)
        dist_in2 = convert(ProbabilityDistribution{V2, GaussianMeanVariance}, msg_in2.dist)

        m_concat, v_concat, dim1, dim2 = mergeInputs(dist_in1, dist_in2)
        dim_tot = dim1 + dim2

        log_prior_pdf(x) = -0.5*(dim_tot*log(2pi) + log(det(v_concat)) + transpose(x-m_concat)*inv(v_concat)*(x-m_concat))

        function log_joint_dims(s::Array, dim1::Int64, dim2::Int64)
            if dim1 == 1
                if dim2 == 1
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1]::Number,s[end]::Number))
                else
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1],s[2:end]))
                end
            else
                if dim2 == 1
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1:dim1],s[end]))
                else
                    return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1:dim1],s[dim1+1:end]))
                end
            end
        end

        log_joint(s) = log_joint_dims(s,dim1,dim2)
        d_log_joint(s) = ForwardDiff.gradient(log_joint, s)

        m_post = gradientOptimization(log_joint, d_log_joint, m_concat, 0.01)
        var_post = Hermitian(inv(- 1.0 .* ForwardDiff.jacobian(d_log_joint, m_post)))

        status[:updated] = true

        mean1, var1, mean2, var2 = decomposePosteriorParameters(dist_in1, dist_in2, m_post, var_post)

        status[:message] = approxMessageBivariate(dist_in1.params[:m],dist_in1.params[:v], mean1, var1)
        return approxMessageBivariate(dist_in2.params[:m],dist_in2.params[:v],mean2,var2)
    end

end

function ruleMBivariateLOutNGG(msg_out::Message{Fout, Vout}, msg_in1::Message{F1, V1}, msg_in2::Message{F2, V2}, g::Function, status::Dict, n_samples::Int) where {Fout<:SoftFactor, Vout<:VariateType, F1<:Gaussian, V1<:VariateType, F2<:Gaussian, V2<:VariateType}

    dist_in1 = convert(ProbabilityDistribution{V1, GaussianMeanVariance}, msg_in1.dist)
    dist_in2 = convert(ProbabilityDistribution{V2, GaussianMeanVariance}, msg_in2.dist)

    m_concat, v_concat, dim1, dim2 = mergeInputs(dist_in1, dist_in2)
    dim_tot = dim1 + dim2

    log_prior_pdf(x) = -0.5*(dim_tot*log(2pi) + log(det(v_concat)) + transpose(x-m_concat)*inv(v_concat)*(x-m_concat))

    function log_joint_dims(s::Array,dim1::Int64,dim2::Int64)
        if dim1 == 1
            if dim2 == 1
                return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1]::Number,s[end]::Number))
            else
                return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1],s[2:end]))
            end
        else
            if dim2 == 1
                return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1:dim1],s[end]))
            else
                return log_prior_pdf(s) + logPdf(msg_out.dist,g(s[1:dim1],s[dim1+1:end]))
            end
        end
    end

    log_joint(s) = log_joint_dims(s,dim1,dim2)
    d_log_joint(s) = ForwardDiff.gradient(log_joint, s)

    m_post = gradientOptimization(log_joint, d_log_joint, m_concat, 0.01)
    var_post = inv(- 1.0 .* ForwardDiff.jacobian(d_log_joint, m_post))

    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m_post, v=var_post)

end

#--------------------------
# Custom inbounds collector
#--------------------------

function collectSumProductNodeInbounds(node::Bivariate{Laplace}, entry::ScheduleEntry)
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry

    inbounds = Any[]
    for node_interface in node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface == entry.interface
            haskey(interface_to_schedule_entry, node_interface) || error("This rule requires the incoming message on the out interface. Try altering execution order to ensure its availability.")
            if entry.message_update_rule == SPBivariateLOutNGG
                push!(inbounds, nothing)
            else
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            end
        elseif isa(inbound_interface.node, Clamp)
            # Hard-code outbound message of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Message))
        else
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    # Push function (and inverse) to calling signature
    # These functions needs to be defined in the scope of the user
    push!(inbounds, Dict{Symbol, Any}(:g => node.g,
                                      :keyword => false))
    status = "currentGraph().nodes[:$(node.id)].status"
    push!(inbounds, Dict{Symbol, Any}(:status => status,
                                      :keyword => false))
    push!(inbounds, node.n_samples)
    return inbounds
end

#--------------------------
# Custom marginal inbounds collector
#--------------------------

function collectMarginalNodeInbounds(node::Bivariate, entry::MarginalEntry)
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

    # Push function and status to calling signature
    # The function needs to be defined in the scope of the user
    push!(inbounds, Dict{Symbol, Any}(:g => node.g,
                                      :keyword => false))
    status = "currentGraph().nodes[:$(node.id)].status"
    push!(inbounds, Dict{Symbol, Any}(:status => status,
                                      :keyword => false))
    push!(inbounds, n_samples)
    return inbounds
end

################################################################################
# Gradient optimization subroutine
################################################################################
function gradientOptimization(log_joint::Function, d_log_joint::Function, m_initial, step_size)
    
    dim_tot = length(m_initial)
    m_total = zeros(dim_tot)
    m_average = zeros(dim_tot)
    m_new = zeros(dim_tot)
    m_old = m_initial
    satisfied = false
    step_count = 0

    while !satisfied
        m_new = m_old .+ step_size.*d_log_joint(m_old)
        if log_joint(m_new) > log_joint(m_old)
            proposal_step_size = 10*step_size
            m_proposal = m_old .+ proposal_step_size.*d_log_joint(m_old)
            if log_joint(m_proposal) > log_joint(m_new)
                m_new = m_proposal
                step_size = proposal_step_size
            end
        else
            step_size = 0.1*step_size
            m_new = m_old .+ step_size.*d_log_joint(m_old)
        end
        step_count += 1
        m_total .+= m_old
        m_average = m_total ./ step_count
        if step_count > 10
            if sum(sqrt.(((m_new.-m_average)./m_average).^2)) < dim_tot*0.1
                satisfied = true
            end
        end
        if step_count > dim_tot*250
            satisfied = true
        end
        m_old = m_new
    end
    return m_new
end


################################################################################
# Helper functions for update rules
################################################################################
function mergeInputs(dist1::ProbabilityDistribution, dist2::ProbabilityDistribution)
    
    m_concat = [dist1.params[:m];dist2.params[:m]]
    dim1 = dims(dist1)
    dim2 = dims(dist2)
    dim_tot = dim1 + dim2
    
    v_concat = zeros(dim_tot, dim_tot)

    if dim1 == 1
        v_concat[dim1,dim1] = dist1.params[:v]
    else
        v_concat[1:dim1,1:dim1] = dist1.params[:v]
    end

    if dim2 == 1
        v_concat[end,end] = dist2.params[:v]
    else
        v_concat[dim1+1:end,dim1+1:end] = dist2.params[:v]
    end

    return (m_concat, v_concat, dim1, dim2)
end

function decomposePosteriorParameters(dist1::ProbabilityDistribution,
    dist2::ProbabilityDistribution, m_post, var_post)

    dim1 = dims(dist1)
    dim2 = dims(dist2)
    
    if dim1 == 1
        mean1 = m_post[1]
        var1 = var_post[1]
    else
        mean1 = m_post[1:dim1]
        var1 = var_post[1:dim1, 1:dim1]
    end

    if dim2 == 1
        mean2 = m_post[end]
        var2 = var_post[end]
    else
        mean2 = m_post[dim1+1:end]
        var2 = var_post[dim1+1:end,dim1+1:end]
    end
    return (mean1, var1, mean2, var2)
end