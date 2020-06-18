export SwitchingGaussian

"""
Description:
A switching Gaussian factor node

    Defines probability distribution over out
    out ∈ R^d,x ∈ R^d
    s is Categorical switch variable
    A is a (array of matrices) tensor of possible matrices
    Q is a (array of PD matrices) tensor of PD matrices for covariance parametrization
    f(out, m,s, A,Q) = N(out|A(s)m,Q(s))


Interfaces:

    1. out
    2. m
    3. s
    4. A
    5. Q


"""
mutable struct SwitchingGaussian <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    A::Vector{Matrix{Float64}}
    Q::Vector{Matrix{Float64}}
    function SwitchingGaussian(out,m,s;A::Vector{Matrix{Float64}},Q::Vector{Matrix{Float64}}, id=generateId(SwitchingGaussian))
        @ensureVariables(out,m,s)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}(),A,Q)
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:s] = self.interfaces[3] = associate!(Interface(self), s)
        return self
    end
end
slug(::Type{SwitchingGaussian}) = "SG"

function averageEnergy(::Type{SwitchingGaussian}, marg_out_mean::ProbabilityDistribution{Multivariate},marg_s::ProbabilityDistribution{Univariate}
                       ,A::Vector{Matrix{Float64}},Q::Vector{Matrix{Float64}})
    (m, V) = unsafeMeanCov(marg_out_mean)
    p = marg_s.params[:p]
    d = Int64(dims(marg_out_mean)/2)
    A_combination = zeros(d,d)
    Q_combination = zeros(d,d)
    q_sum = 0.0
    for i=1:length(p)
        A_combination += p[i]*A[i]
        Q_combination += p[i]*inv(Q[i])
        q_sum += p[i]*log(det(Q[i]))
    end

    0.5*d*log(2*pi) -
    0.5*q_sum +
    0.5*tr( Q_combination*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ) )
end
