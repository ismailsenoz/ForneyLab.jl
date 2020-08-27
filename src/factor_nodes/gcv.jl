export GCV, GaussHermite, SphericalRadial, Laplace

abstract type ApproximationMethod end
abstract type GaussHermite <: ApproximationMethod end
abstract type SphericalRadial <: ApproximationMethod end
abstract type Laplace <: ApproximationMethod end

struct MultivariateGaussianMeanVariance end

vague(::Type{<:MultivariateGaussianMeanVariance}) = ProbabilityDistribution(Multivariate, GaussianMeanVariance, m = zeros(2), v = diageye(2))

mutable struct GCV{T<:ApproximationMethod} <: SoftFactor
    id::Symbol
    interfaces::Array{Interface}
    i::Dict{Symbol, Interface}
    g::Function # Matrix valued positive definite function that expresses the output vector as a function of the input.

    function GCV{GaussHermite}(out, m, z, g::Function; id=ForneyLab.generateId(GCV{GaussHermite}))
        @ensureVariables(out, m, z)
        self = new(id, Vector{Interface}(undef, 3), Dict{Symbol,Interface}(), g)
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)

        return self
    end

    function GCV{SphericalRadial}(out, m, z, g::Function; id=ForneyLab.generateId(GCV{SphericalRadial}))
        @ensureVariables(out, m, z)
        self = new(id, Vector{Interface}(undef, 3), Dict{Symbol,Interface}(), g)
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)

        return self
    end

    function GCV{Laplace}(out, m, z, g::Function; id=ForneyLab.generateId(GCV{Laplace}))
        @ensureVariables(out, m, z)
        self = new(id, Vector{Interface}(undef, 3), Dict{Symbol,Interface}(), g)
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)

        return self
    end
end

function GCV(out, in1, g::Function; id=ForneyLab.generateId(GCV{GaussHermite}))
    return GCV{GaussHermite}(out, m, z, g::Function;id=id)
end

slug(::Type{GCV}) = "GCV"

function averageEnergy(Node::Type{GCV{GaussHermite}}, marg_out_mean::ProbabilityDistribution{Multivariate, F1}, marg_z::ProbabilityDistribution{Multivariate, F2}, g::Function) where { F1 <:Gaussian, F2 <:Gaussian }
    (m, V) = unsafeMeanCov(marg_out_mean)
    (mz,Vz) = unsafeMeanCov(marg_z)

    d = Int64(dims(marg_out_mean) / 2)

    cubature = ghcubature(d, 20)
    weights = getweights(cubature)
    points  = getpoints(cubature, mz, Vz)

    gs = Base.Generator(points) do point
        return g(point)
    end

    Λ_out = mapreduce(t -> t[1] * cholinv(t[2]), +, zip(weights, gs))
    log_det_sum = mapreduce(t -> t[1] * logdet(t[2]), +, zip(weights, gs))

    @views 0.5*d*log(2*pi) +
    0.5*log_det_sum +
    0.5*tr( Λ_out*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ) )
end

function averageEnergy(Node::Type{GCV{SphericalRadial}}, marg_out_mean::ProbabilityDistribution{Multivariate, F1}, marg_z::ProbabilityDistribution{Multivariate, F2}, g::Function) where { F1 <:Gaussian, F2 <:Gaussian }
    (m, V) = unsafeMeanCov(marg_out_mean)
    (mz,Vz) = unsafeMeanCov(marg_z)
    d = Int64(dims(marg_out_mean) / 2)

    cubature = srcubature(d)
    weights  = getweights(cubature)
    points   = getpoints(cubature, mz, Vz)

    gs = Base.Generator(points) do point
        return g(point)
    end

    Λ_out = mapreduce(t -> t[1] * cholinv(t[2]), +, zip(weights, gs))
    log_det_sum = mapreduce(t -> t[1] * logdet(t[2]), +, zip(weights, gs))

    @views 0.5*d*log(2*pi) +
    0.5*log_det_sum +
    0.5*tr( Λ_out*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ) )
end

averageEnergy(Node::Type{GCV{Laplace}}, marg_out_mean::ProbabilityDistribution{Multivariate, F1}, marg_z::ProbabilityDistribution{Multivariate, F2}, g::Function) where { F1 <:Gaussian, F2 <:Gaussian } =
    averageEnergy(GCV{SphericalRadial}, marg_out_mean, marg_z, g)

function collectAverageEnergyInbounds(node::GCV)
    inbounds = Any[]
    local_posterior_factor_to_region = localPosteriorFactorToRegion(node)

    encountered_posterior_factors = Union{PosteriorFactor, Edge}[] # Keep track of encountered posterior factors
    for node_interface in node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_posterior_factor = posteriorFactor(node_interface.edge)

        if (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(copy(inbound_interface.node), ProbabilityDistribution)) # Copy Clamp before assembly to prevent overwriting dist_or_msg field
        elseif !(current_posterior_factor in encountered_posterior_factors)
            # Collect marginal entry from marginal dictionary (if marginal entry is not already accepted)
            target = local_posterior_factor_to_region[current_posterior_factor]
            push!(inbounds, current_inference_algorithm.target_to_marginal_entry[target])
        end

        push!(encountered_posterior_factors, current_posterior_factor)
    end


    push!(inbounds, Dict{Symbol, Any}(:g => node.g,
                                      :keyword => false))

    return inbounds
end
