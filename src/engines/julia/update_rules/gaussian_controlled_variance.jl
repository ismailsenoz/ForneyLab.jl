export
ruleSVBGaussianControlledVarianceOutNGDDD,
ruleSVBGaussianControlledVarianceXGNDDD,
ruleSVBGaussianControlledVarianceZDNDD,
ruleSVBGaussianControlledVarianceΚDDND,
ruleSVBGaussianControlledVarianceΩDDDN,
ruleMGaussianControlledVarianceGGDDD,
ruleSVBGaussianControlledVarianceOutNGDD,
ruleSVBGaussianControlledVarianceXGNDD,
ruleSVBGaussianControlledVarianceZDGGD,
ruleSVBGaussianControlledVarianceΚDGGD,
ruleSVBGaussianControlledVarianceΩDDN,
ruleMGaussianControlledVarianceGGDD,
ruleMGaussianControlledVarianceDGGD

function ruleSVBGaussianControlledVarianceOutNGDDD(dist_out::Nothing,
                                                   msg_x::Message{F, Univariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_x.dist)
    m_x = dist_x.params[:m]
    v_x = dist_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)


    return Message(Univariate, GaussianMeanVariance, m=m_x, v=v_x+inv(A*B))
end


function ruleSVBGaussianControlledVarianceOutNGDD(dist_out::Nothing,
                                                   msg_x::Message{F1, Univariate},
                                                   dist_z_κ::ProbabilityDistribution{Multivariate, F2},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian}

    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_x.dist)
    dist_z_κ = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance},dist_z_κ)
    m_x = dist_x.params[:m]
    v_x = dist_x.params[:v]
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    A = exp(-m_ω+v_ω/2)
    B = quadratureExpectationExp(dist_z_κ,30)


    return Message(Univariate, GaussianMeanVariance, m=m_x, v=v_x+inv(A*B))
end

function ruleSVBGaussianControlledVarianceXGNDDD(msg_out::Message{F, Univariate},
                                                   dist_x::Nothing,
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
    m_out = dist_out.params[:m]
    v_out = dist_out.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, GaussianMeanVariance, m=m_out, v=v_out+inv(A*B))
end


function ruleSVBGaussianControlledVarianceXGNDD(msg_out::Message{F1, Univariate},
                                                   dist_x::Nothing,
                                                   dist_z_κ::ProbabilityDistribution{Multivariate, F2},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian}

    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
    dist_z_κ = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance},dist_z_κ)
    m_out = dist_out.params[:m]
    v_out = dist_out.params[:v]
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    A = exp(-m_ω+v_ω/2)
    B = quadratureExpectationExp(dist_z_κ,30)

    return Message(Univariate, GaussianMeanVariance, m=m_out, v=v_out+inv(A*B))
end


function ruleSVBGaussianControlledVarianceZDNDD(dist_out_x::ProbabilityDistribution{Multivariate, F},
                                                dist_z::Nothing,
                                                dist_κ::ProbabilityDistribution{Univariate},
                                                dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)

    return Message(Univariate, ExponentialLinearQuadratic, a=m_κ, b=Psi*A,c=-m_κ,d=v_κ)
end

#
function ruleSVBGaussianControlledVarianceZDGGD(dist_out_x::ProbabilityDistribution{Multivariate,F1},
                                                msg_z::Message{F2,Univariate},
                                                msg_κ::Message{F3,Univariate},
                                                dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian,F3<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(msg_κ.dist)
    m_z, v_z = unsafeMeanCov(msg_z.dist)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    h(x) = -0.5*((x[1]-m_κ)^2/v_κ +(x[2]-m_z)^2/v_z + x[1]*x[2] + A*Psi*exp(-x[1]*x[2]))
    newton_m, newton_v = NewtonMethod(h,[m_κ; m_z],10)
    mean = newton_m[2] + newton_v[1,2]*inv(newton_v[1,1])*(m_κ-newton_m[1])
    var = newton_v[2,2] - newton_v[1,2]*inv(newton_v[1,1])*newton_v[1,2] + (newton_v[1,2]*inv(newton_v[1,1]))^2*v_κ


    return Message(Univariate, GaussianMeanVariance, m=mean ,v=var)
end


function ruleSVBGaussianControlledVarianceΚDDND(dist_out_x::ProbabilityDistribution{Multivariate, F},
                                                dist_z::ProbabilityDistribution{Univariate},
                                                dist_κ::Nothing,
                                                dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)

    return Message(Univariate, ExponentialLinearQuadratic, a=m_z, b=Psi*A,c=-m_z,d=v_z)
end

#
function ruleSVBGaussianControlledVarianceΚDGGD(dist_out_x::ProbabilityDistribution{Multivariate,F1},
                                                msg_z::Message{F2,Univariate},
                                                msg_κ::Message{F3,Univariate},
                                                dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian,F3<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(msg_κ.dist)
    m_z, v_z = unsafeMeanCov(msg_z.dist)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    h(x) = -0.5*((x[1]-m_κ)^2/v_κ +(x[2]-m_z)^2/v_z + x[1]*x[2] + A*Psi*exp(-x[1]*x[2]))
    newton_m, newton_v = NewtonMethod(h,[m_κ; m_z],10)
    mean = newton_m[1] + newton_v[1,2]*inv(newton_v[2,2])*(m_κ-newton_m[2])
    var = newton_v[1,1] - newton_v[1,2]*inv(newton_v[2,2])*newton_v[1,2] + (newton_v[1,2]*inv(newton_v[2,2]))^2*v_z


    return Message(Univariate, GaussianMeanVariance, m=mean ,v=var)
end

function ruleSVBGaussianControlledVarianceΩDDDN(dist_out_x::ProbabilityDistribution{Multivariate, F},
                                                dist_z::ProbabilityDistribution{Univariate},
                                                dist_κ::ProbabilityDistribution{Univariate},
                                                dist_ω::Nothing) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, ExponentialLinearQuadratic, a=1.0, b=Psi*B,c=-1.0,d=0.0)
end

function ruleSVBGaussianControlledVarianceΩDDN(dist_out_x::ProbabilityDistribution{Multivariate, F1},
                                               dist_z_κ::ProbabilityDistribution{Multivariate, F2},
                                               dist_ω::Nothing) where {F1<:Gaussian, F2<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    dist_z_κ = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z_κ)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]


    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    B = quadratureExpectationExp(dist_z_κ,30)

    return Message(Univariate, ExponentialLinearQuadratic, a=1.0, b=Psi*B,c=-1.0,d=0.0)
end

function ruleMGaussianControlledVarianceGGDDD(msg_out::Message{F1, Univariate},
                                              msg_x::Message{F2, Univariate},
                                              dist_z::ProbabilityDistribution{Univariate},
                                              dist_κ::ProbabilityDistribution{Univariate},
                                              dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian, F2 <: Gaussian}
    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_out.dist)
    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_x.dist)
    m_x = dist_x.params[:m]
    w_x = dist_x.params[:w]
    m_out = dist_out.params[:m]
    w_out = dist_out.params[:w]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)
    W = [w_out+A*B -A*B; -A*B w_x+A*B]
    m = inv(W)*[m_out*w_out; m_x*w_x]

    return ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m, w=W)

end

function ruleMGaussianControlledVarianceGGDD(msg_out::Message{F1, Univariate},
                                              msg_x::Message{F2, Univariate},
                                              dist_z_κ::ProbabilityDistribution{Multivariate, F3},
                                              dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian, F2 <: Gaussian, F3<:Gaussian}
    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_out.dist)
    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_x.dist)
    dist_z_κ = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_z_κ)
    m_x = dist_x.params[:m]
    w_x = dist_x.params[:w]
    m_out = dist_out.params[:m]
    w_out = dist_out.params[:w]
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    A = exp(-m_ω+v_ω/2)
    B = quadratureExpectationExp(dist_z_κ,30)

    W = [w_out+A*B -A*B; -A*B w_x+A*B]
    m = inv(W)*[m_out*w_out; m_x*w_x]

    return ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m, w=W)

end


function ruleMGaussianControlledVarianceDGGD(dist_out_x::ProbabilityDistribution{Multivariate, F1},
                                             msg_z::Message{F2, Univariate},
                                             msg_κ::Message{F3, Univariate},
                                             dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(msg_z.dist)
    m_κ, v_κ = unsafeMeanCov(msg_κ.dist)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    h(x) = -0.5*((x[1]-m_κ)^2/v_κ +(x[2]-m_z)^2/v_z + x[1]*x[2] + A*Psi*exp(-x[1]*x[2]))
    newton_m, newton_v = NewtonMethod(h,[m_κ; m_z],10)
    # dist = ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=newton_m,v=newton_v)*ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=[m_κ;m_z], v=[v_κ 0.0;0.0 v_z])

    return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=newton_m,v=newton_v)
end


# ###Custom inbounds
function collectStructuredVariationalNodeInbounds(node::GaussianControlledVariance, entry::ScheduleEntry, interface_to_msg_idx::Dict{Interface, Int})
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
            if entry.msg_update_rule in [SVBGaussianControlledVarianceΚDGGD, SVBGaussianControlledVarianceZDGGD]
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
