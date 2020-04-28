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
    Λ_out = sum([weights_m[k+1]*inv(g_sigma[k+1]) for k=0:2*d])

    return Message(Multivariate, GaussianMeanVariance,m=mean_m,v=cov_m+inv(Λ_out))
end

function ruleSVBGCVCubatureMGND(msg_out::Message{F, Multivariate},msg_m::Nothing, dist_z::ProbabilityDistribution{Multivariate}, g::Function) where F<:Gaussian
    d = dims(msg_out.dist)
    mean_out, cov_out = unsafeMeanCov(msg_out.dist)
    mean_z, cov_z = unsafeMeanCov(dist_z)
    (sigma_points, weights_m, weights_c) = ForneyLab.sigmaPointsAndWeights(mean_z,cov_z)
    # Unscented approximation
    g_sigma = g.(sigma_points)
    Λ_m = sum([weights_m[k+1]*inv(g_sigma[k+1]) for k=0:2*d])

    return Message(Multivariate, GaussianMeanVariance,m=mean_out,v=cov_out+inv(Λ_m))
end

function ruleSVBGCVCubatureZDN(dist_out_mean::ProbabilityDistribution{Multivariate}, msg_z::Nothing, g::Function)
    d = Int64(dims(dist_out_mean)/2)
    m_out_mean, cov_out_mean = unsafeMeanCov(dist_out_mean)
    psi = cov_out_mean[1:d,1:d] - cov_out_mean[1:d,d+1:end] - cov_out_mean[d+1:end, 1:d] + cov_out_mean[d+1:end,d+1:end] + (m_out_mean[1:d] - m_out_mean[d+1:end])*(m_out_mean[1:d] - m_out_mean[d+1:end])'

    l_pdf(z) = -0.5*(logdet(g(z)) + tr(inv(g(z))*psi))
    return Message(Multivariate, Function, log_pdf=l_pdf)
end


function ruleMGCVCubatureMGGD(msg_out::Message{F1, Multivariate},msg_m::Message{F2, Multivariate},dist_z::ProbabilityDistribution{Multivariate,F3},g::Function) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}
    d = dims(msg_out.dist)
    xi_out,Λ_out = unsafeWeightedMeanPrecision(msg_out.dist)
    xi_mean,Λ_m = unsafeWeightedMeanPrecision(msg_m.dist)
    dist_z = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z)
    h(z) = inv(g(z))
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

function multiDimensionalPointsWeights(n::Int64,p::Int64)
    sigma_points = [-13.406487338144908, -12.823799749487808, -12.342964222859672, -11.915061943114164, -11.52141540078703, -11.152404385585125, -10.802260753684713, -10.467185421342812, -10.144509941292846, -9.832269807777967, -9.528965823390115, -9.233420890219161, -8.944689217325474, -8.661996168134518, -8.384696940416266, -8.112247311162792, -7.84418238446082, -7.580100807857489, -7.319652822304534, -7.062531060248865, -6.808463352858795, -6.557207031921539, -6.308544361112134, -6.062278832614302, -5.818232135203517, -5.576241649329924, -5.33615836013836, -5.097845105089136, -4.8611750917912095, -4.6260306357871555, -4.392302078682683, -4.15988685513103, -3.9286886834276706, -3.6986168593184914, -3.469585636418589, -3.241513679631013, -3.014323580331155, -2.787941423981989, -2.5622964023726076, -2.3373204639068783, -2.112947996371188, -1.8891155374270083, -1.6657615087415094, -1.4428259702159327, -1.2202503912189528, -0.9979774360981053, -0.7759507615401456, -0.5541148235916169, -0.3324146923422318, -0.11079587242243949, 0.11079587242243949, 0.3324146923422318, 0.5541148235916169, 0.7759507615401456, 0.9979774360981053, 1.2202503912189528, 1.4428259702159327, 1.6657615087415094, 1.8891155374270083, 2.112947996371188, 2.3373204639068783, 2.5622964023726076, 2.787941423981989, 3.014323580331155, 3.241513679631013, 3.469585636418589, 3.6986168593184914, 3.9286886834276706, 4.15988685513103, 4.392302078682683, 4.6260306357871555, 4.8611750917912095, 5.097845105089136, 5.33615836013836, 5.576241649329924, 5.818232135203517, 6.062278832614302, 6.308544361112134, 6.557207031921539, 6.808463352858795, 7.062531060248865, 7.319652822304534, 7.580100807857489, 7.84418238446082, 8.112247311162792, 8.384696940416266, 8.661996168134518, 8.944689217325474, 9.233420890219161, 9.528965823390115, 9.832269807777967, 10.144509941292846, 10.467185421342812, 10.802260753684713, 11.152404385585125, 11.52141540078703, 11.915061943114164, 12.342964222859672, 12.823799749487808, 13.406487338144908]
    sigma_weights = [5.90806786475396e-79, 1.972860574879216e-72, 3.083028990003297e-67, 9.019222303693804e-63, 8.518883081761774e-59, 3.459477936475577e-55, 7.191529463463525e-52, 8.597563954825022e-49, 6.4207252053483165e-46, 3.1852178778359564e-43, 1.1004706827141981e-40, 2.7487848843571714e-38, 5.1162326043853164e-36, 7.274572596887586e-34, 8.067434278709346e-32, 7.101812226384877e-30, 5.037791166213212e-28, 2.917350072629348e-26, 1.3948415260687509e-24, 5.561026961659241e-23, 1.864997675130272e-21, 5.302316183131963e-20, 1.28683292112117e-18, 2.6824921647603466e-17, 4.829835321703033e-16, 7.548896877915255e-15, 1.0288749373509815e-13, 1.2278785144101149e-12, 1.2879038257315609e-11, 1.1913006349290596e-10, 9.747921253871486e-10, 7.075857283889495e-9, 4.5681275084849026e-8, 2.6290974837537006e-7, 1.3517971591103645e-6, 6.221524817777747e-6, 2.5676159384548995e-5, 9.517162778551009e-5, 0.00031729197104329556, 0.0009526921885486135, 0.002579273260059073, 0.006303000285608099, 0.013915665220231849, 0.027779127385933522, 0.05017581267742825, 0.08205182739122392, 0.12153798684410465, 0.16313003050278302, 0.1984628502541864, 0.21889262958743966, 0.21889262958743966, 0.1984628502541864, 0.16313003050278302, 0.12153798684410465, 0.08205182739122392, 0.05017581267742825, 0.027779127385933522, 0.013915665220231849, 0.006303000285608099, 0.002579273260059073, 0.0009526921885486135, 0.00031729197104329556, 9.517162778551009e-5, 2.5676159384548995e-5, 6.221524817777747e-6, 1.3517971591103645e-6, 2.6290974837537006e-7, 4.5681275084849026e-8, 7.075857283889495e-9, 9.747921253871486e-10, 1.1913006349290596e-10, 1.2879038257315609e-11, 1.2278785144101149e-12, 1.0288749373509815e-13, 7.548896877915255e-15, 4.829835321703033e-16, 2.6824921647603466e-17, 1.28683292112117e-18, 5.302316183131963e-20, 1.864997675130272e-21, 5.561026961659241e-23, 1.3948415260687509e-24, 2.917350072629348e-26, 5.037791166213212e-28, 7.101812226384877e-30, 8.067434278709346e-32, 7.274572596887586e-34, 5.1162326043853164e-36, 2.7487848843571714e-38, 1.1004706827141981e-40, 3.1852178778359564e-43, 6.4207252053483165e-46, 8.597563954825022e-49, 7.191529463463525e-52, 3.459477936475577e-55, 8.518883081761774e-59, 9.019222303693804e-63, 3.083028990003297e-67, 1.972860574879216e-72, 5.90806786475396e-79]
    # sigma_points, sigma_weights = gausshermite(p)
    points_iter = product(repeat([sigma_points],n)...)
    weights_iter = product(repeat([sigma_weights],n)...)
    return points_iter, weights_iter
end

# using FastGaussQuadrature

function gaussHermiteCubature(g::Function,dist::ProbabilityDistribution{Multivariate,GaussianMeanVariance},p::Int64)
    d = dims(dist)
    m, P = ForneyLab.unsafeMeanCov(dist)
    sqrtP = sqrt(P)
    points_iter, weights_iter = multiDimensionalPointsWeights(d,p)
    #compute normalization constant
    normalization = 0.0
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        normalization += weight.*g(m+ sqrt(2).*sqrtP*point)
    end
    normalization = normalization/sqrt(pi)
    #compute mean
    mean = zeros(d)
    h(z) = g(z).* z ./normalization
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        mean += weight.*h(m + sqrt(2).*sqrtP*point)
    end
    mean = mean./sqrt(pi)
    cov = zeros(d,d)
    f(z) = (g(z)/normalization) .* (z-mean)*(z-mean)'
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        cov += weight.*f(m + sqrt(2).*sqrtP*point)
    end
    cov = cov./sqrt(pi)
    return mean,cov
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
        x = x_0 - inv(hessian)*grad
        x_0 = x
    end
    var = inv(ForwardDiff.hessian(g,x))
    return x, var
end

function kernelExpectation(g::Function,dist::ProbabilityDistribution{Multivariate,GaussianMeanVariance},p::Int64)
    d = dims(dist)
    m, P = ForneyLab.unsafeMeanCov(dist)
    sqrtP = sqrt(P)
    points_iter, weights_iter = multiDimensionalPointsWeights(d,p)
    #compute normalization constant
    g_bar = zeros(d,d)
    for (point_tuple, weights) in zip(points_iter, weights_iter)
        weight = prod(weights)
        point = collect(point_tuple)
        g_bar += weight.*g(m+sqrt(2).*sqrtP*point)
    end
    return g_bar./sqrt(pi)
end
