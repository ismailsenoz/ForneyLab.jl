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
ruleSVBGaussianControlledVarianceOutNEDDD,
ruleSVBGaussianControlledVarianceMENDDD,
ruleMGaussianControlledVarianceGGDD,
ruleMGaussianControlledVarianceDGGD,
ruleMGaussianControlledVarianceEGDDD,
ruleMGaussianControlledVarianceGEDDD,
prod!
#
# function ruleSVBGaussianControlledVarianceMENDDD(msg_out::Message{ExponentialLinearQuadratic},
#                                                    msg_x::Message{F, Univariate},
#                                                    dist_z::ProbabilityDistribution{Univariate},
#                                                    dist_κ::ProbabilityDistribution{Univariate},
#                                                    dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian
#
#     dist_out = msg_out.dist
#     dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_x.dist)
#     approx_dist = dist_x*dist_out
#     approx_msg = Message(Univariate,GaussianMeanVariance,m=approx_dist.params[:m],v=approx_dist.params[:v])
#
#     return ruleSVBGaussianControlledVarianceOutNGDDD(nothing, approx_msg,dist_z,dist_κ,dist_ω)
# end
#
# function ruleSVBGaussianControlledVarianceOutNEDDD(msg_out::Message{F, Univariate},
#                                                    msg_x::Message{ExponentialLinearQuadratic},
#                                                    dist_z::ProbabilityDistribution{Univariate},
#                                                    dist_κ::ProbabilityDistribution{Univariate},
#                                                    dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian
#
#
#    dist_x = msg_x.dist
#    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
#    approx_dist = dist_x*dist_out
#    approx_msg = Message(Univariate,GaussianMeanVariance,m=approx_dist.params[:m],v=approx_dist.params[:v])
#
#    return ruleSVBGaussianControlledVarianceOutNGDDD(nothing, approx_msg,dist_z,dist_κ,dist_ω)
# end
function ruleSVBGaussianControlledVarianceMENDDD(msg_out::Message{ExponentialLinearQuadratic},
                                                   msg_x::Message{F, Univariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    # msg_out_prime = approximateDoubleExp(msg_out)
    # return ruleSVBGaussianControlledVarianceOutNGDDD(nothing,msg_out_prime,dist_z,dist_κ,dist_ω)
    msg_out_prime = ruleSVBGaussianControlledVarianceOutNGDDD(nothing,msg_x,dist_z,dist_κ,dist_ω)
    approx_dist = msg_out_prime.dist*msg_out.dist
    approx_msg = Message(GaussianMeanVariance,m=approx_dist.params[:m],v=approx_dist.params[:v])
    return ruleSVBGaussianControlledVarianceOutNGDDD(nothing, approx_msg,dist_z,dist_κ,dist_ω)
end

function ruleSVBGaussianControlledVarianceOutNEDDD(msg_out::Message{F, Univariate},
                                                   msg_x::Message{ExponentialLinearQuadratic},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    # msg_mean_prime = approximateDoubleExp(msg_mean)
    # return ruleSVBGaussianControlledVarianceOutNGDDD(nothing,msg_mean_prime,dist_z,dist_κ,dist_ω)
    msg_x_prime = ruleSVBGaussianControlledVarianceOutNGDDD(nothing,msg_out,dist_z,dist_κ,dist_ω)
    approx_dist = msg_x_prime.dist*msg_x.dist
    approx_msg = Message(GaussianMeanVariance,m=approx_dist.params[:m],v=approx_dist.params[:v])
    return ruleSVBGaussianControlledVarianceOutNGDDD(nothing, approx_msg,dist_z,dist_κ,dist_ω)
end

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
    B = quadratureExpectationExp(dist_z_κ,10)

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
    W = [w_out+A*B -A*B; -A*B w_x+A*B] +1e-8diageye(2)
    m = inv(W)*[m_out*w_out; m_x*w_x]

    return ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m, w=W)

end
# function ruleMGaussianControlledVarianceEGDDD(msg_out::Message{ExponentialLinearQuadratic},
#                                               msg_x::Message{F1, Univariate},
#                                               dist_z::ProbabilityDistribution{Univariate},
#                                               dist_κ::ProbabilityDistribution{Univariate},
#                                               dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian}
#     msg_out_prime = approximateDoubleExp(msg_out)
#
#     return ruleMGaussianControlledVarianceGGDDD(msg_out_prime,msg_x,dist_z,dist_κ,dist_ω)
#
# end
#
#
# function ruleMGaussianControlledVarianceGEDDD(msg_out::Message{F1, Univariate},
#                                               msg_x::Message{ExponentialLinearQuadratic},
#                                               dist_z::ProbabilityDistribution{Univariate},
#                                               dist_κ::ProbabilityDistribution{Univariate},
#                                               dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian}
#
#
#
#     msg_x_prime = approximateDoubleExp(msg_x)
#
#     return ruleMGaussianControlledVarianceGGDDD(msg_out,msg_x_prime,dist_z,dist_κ,dist_ω)
#
# end
# function ruleMGaussianControlledVarianceEGDDD(msg_out::Message{ExponentialLinearQuadratic},
#                                               msg_x::Message{F1, Univariate},
#                                               dist_z::ProbabilityDistribution{Univariate},
#                                               dist_κ::ProbabilityDistribution{Univariate},
#                                               dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian}
#     m_x, v_x = unsafeMeanCov(msg_x.dist)
#     m_z, v_z = unsafeMeanCov(dist_z)
#     m_κ, v_κ = unsafeMeanCov(dist_κ)
#     m_ω, v_ω = unsafeMeanCov(dist_ω)
#     ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
#     A = exp(-m_ω+v_ω/2)
#     B = exp(-m_κ*m_z + ksi/2)
#     a = msg_out.dist.params[:a]
#     b = msg_out.dist.params[:b]
#     c = msg_out.dist.params[:c]
#     d = msg_out.dist.params[:d]
#
#     g(x) = exp(-0.5*(a*x[1]+b*exp(c*x[1] + d*x[1]^2/2)+(x[2]-m_x)^2/v_x + (x[1]-x[2])^2/(A*B)))
#     m,Σ = multivariateNormalApproximation(g,[-20; -20],[20; 20.])
#
#     return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=m,v=Σ+1e-8*diageye(2))
#
# end
#
function ruleMGaussianControlledVarianceEGDDD(msg_out::Message{ExponentialLinearQuadratic},
                                              msg_x::Message{F1, Univariate},
                                              dist_z::ProbabilityDistribution{Univariate},
                                              dist_κ::ProbabilityDistribution{Univariate},
                                              dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian}
    m_x, v_x = unsafeMeanCov(msg_x.dist)
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)
    a = msg_out.dist.params[:a]
    b = msg_out.dist.params[:b]
    c = msg_out.dist.params[:c]
    d = msg_out.dist.params[:d]

    g(x) = a*x[1]+b*exp(c*x[1] + d*x[1]^2/2)+(x[2]-m_x)^2/v_x + (x[1]-x[2])^2/(A*B)
    msg_out_prime = approximateDoubleExp(msg_out)
    x0 = [msg_out_prime.dist.params[:m]; m_x]
    m,Σ = NewtonMethod(g,x0,1)

    return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=m,v=Σ)

end
#
#
# function ruleMGaussianControlledVarianceGEDDD(msg_out::Message{F1, Univariate},
#                                               msg_x::Message{ExponentialLinearQuadratic},
#                                               dist_z::ProbabilityDistribution{Univariate},
#                                               dist_κ::ProbabilityDistribution{Univariate},
#                                               dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian}
#
#
#     m_out, v_out = unsafeMeanCov(msg_out.dist)
#     m_z, v_z = unsafeMeanCov(dist_z)
#     m_κ, v_κ = unsafeMeanCov(dist_κ)
#     m_ω, v_ω = unsafeMeanCov(dist_ω)
#     ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
#     A = exp(-m_ω+v_ω/2)
#     B = exp(-m_κ*m_z + ksi/2)
#     a = msg_x.dist.params[:a]
#     b = msg_x.dist.params[:b]
#     c = msg_x.dist.params[:c]
#     d = msg_x.dist.params[:d]
#
#     g(x) = a*x[2]+b*exp(c*x[2] + d*x[2]^2/2)+(x[1]-m_out)^2/v_out + (x[1]-x[2])^2/(A*B)
#     msg_x_prime = approximateDoubleExp(msg_x)
#     x0 = [m_out;msg_x_prime.dist.params[:m]]
#     m,Σ = NewtonMethod(g,x0,10)
#
#     return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=m,v=Σ)
#
# end

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
    B = quadratureExpectationExp(dist_z_κ,10)

    W = [w_out+A*B -A*B; -A*B w_x+A*B] +1e-8diageye(2)
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

function collectStructuredVariationalNodeInbounds(::GaussianControlledVariance, entry::ScheduleEntry)
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
            if entry.msg_update_rule in [SVBGaussianControlledVarianceΚDGGD,SVBGaussianControlledVarianceZDGGD,SVBGaussianControlledVarianceOutNEDDD,SVBGaussianControlledVarianceMENDDD]
                inbound_idx = interface_to_msg_idx[inbound_interface]
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
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

    return inbounds
end

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff
using Cubature, HCubature

# function kldiv(d1::ProbabilityDistribution,d2::ProbabilityDistribution)
#     (m1,v1) = unsafeMeanCov(d1)
#     (m2,v2) = unsafeMeanCov(d2)
#
#     return 0.5*log(v2/v1) + (v1+(m1-m2)^2)/(2*v2) - 0.5
#
# end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate, F1},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     x_min = m_y-100.0
#     x_max = m_y+100
#
#     g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))*exp(-0.5*(x-m_y)^2/v_y)
#     normalization = hquadrature(g,x_min,x_max)[1]
#     f(x) = x.* g(x)./ normalization
#     m = hquadrature(f,x_min,x_max)[1]
#     h(x) = (x.- m).^2 .* g(x)./ normalization
#     v = hquadrature(h,x_min,x_max)[1]
#
#
#     z.params[:m] = m
#     z.params[:v] = v
#
#     return z
# end

function approximateDoubleExp(msg::Message{ExponentialLinearQuadratic})
    a = msg.dist.params[:a]
    b = msg.dist.params[:b]
    c = msg.dist.params[:c]
    d = msg.dist.params[:d]
    g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
    norm_const = HCubature.hquadrature(g,-100.0,100.0)[1]
    f(x) = x.* g(x)./ norm_const
    m = HCubature.hquadrature(f,-100.0,100.0)[1]
    h(x) = (x.- m).^2 .* g(x)./ norm_const
    v = HCubature.hquadrature(h,-100.0,100.0)[1]

    return Message(GaussianMeanVariance,m=m,v=v)
end

@symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
                y::ProbabilityDistribution{Univariate, F1},
                z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian

    dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
    m_y, v_y = unsafeMeanCov(dist_y)
    a = x.params[:a]
    b = x.params[:b]
    c = x.params[:c]
    d = x.params[:d]
    p = 20

    g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
    normalization_constant = quadrature(g,dist_y,p)
    t(x) = x*g(x)/normalization_constant
    mean_post = quadrature(t,dist_y,p)
    s(x) = (x-mean_post)^2*g(x)/normalization_constant
    var_post = quadrature(s,dist_y,p)


    z.params[:m] = mean_post
    z.params[:v] = var_post


    return z
end


# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate, F1},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#
#     g_der(x) = -0.5*(a + b*(c+d*x)*exp(c*x+d*x^2/2)+2*(x-m_y)/v_y)
#     g_hes(x) = -0.5*(exp(c*x+d*x^2/2)*(b*d+b*(c+d*x)^2)+ 2/v_y)
#     mean_post = m_y
#     for i=1:100
#         @show mean_post = mean_post - g_der(mean_post)/g_hes(mean_post)
#     end
#     # println("derivative ", g_der(mean_post)," ","hessian ", g_hes(mean_post))
#
#     @show z.params[:m] = mean_post
#     @show z.params[:v] = -2/g_hes(mean_post)
#
#
#     return z
# end


#
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y = dist_y.params[:m]
#     v_y = dist_y.params[:v]
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     g(x) = -0.5*(a*x+b*exp(c*x+d*x^2/2)+(x-m_y)^2/v_y)
#     h(x) = ForwardDiff.derivative(g,x)
#     # var_z = 0.0
#     # mean_z = 0.0
#     # for i=1:100
#     gradient = ForwardDiff.derivative(g, m_y)
#     hessian = ForwardDiff.derivative(h, m_y)
#     var_z = -inv(hessian)
#     mean_z = m_y + var_z*gradient
#     m_y = mean_z
#     # end
#     println(mean_z, " ", var_z)
#     z.params[:m] = mean_z
#     z.params[:v] = var_z
#
#     return z
# end

#
# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F<:Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y = dist_y.params[:m]
#     v_y = dist_y.params[:v]
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#     f(x) = exp(c*x+d*x^2/2)
#     g_der(x) = -0.5*(a + b*f(x)*(c+d*x)+2*(x-m_y)/v_y)
#     g_hes(x) = -0.5*(f(x)*b*(a^2-2*a*x*d+x^2*d+d) + 2/v_y)
#     # mean_z = 0.0
#     # for i = 1:10
#     println(g_der(m_y), g_hes(m_y))
#     mean_z = 0.0
#     for i=1:100
#         mean_z = m_y - 0.1*g_der(m_y)
#         m_y = mean_z
#     end
#     # end
#     z.params[:m] = mean_z
#     z.params[:v] = -1/g_hes(mean_z)
#
#
#
#     return z
# end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, ExponentialLinearQuadratic},
#                 y::ProbabilityDistribution{Univariate,F},
#                 z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F <: Gaussian
#
#     dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance}, y)
#     m_y, v_y = unsafeMeanCov(dist_y)
#     a = x.params[:a]
#     b = x.params[:b]
#     c = x.params[:c]
#     d = x.params[:d]
#
#     g(x) = exp(-0.5*(a*x+b*exp(c*x+d*x^2/2)))
#
#     samples = m_y .+ sqrt(v_y) .* randn(1000)
#     mean = sum(g.(samples) ./ sum(g.(samples)) .* samples)
#     var = sum(g.(samples) ./ sum(g.(samples)) .* (samples.-mean).^2)
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end


function quadrature(g::Function,d::ProbabilityDistribution{Univariate,GaussianMeanVariance},p::Int64)
    sigma_points, sigma_weights = gausshermite(p)
    m, v = ForneyLab.unsafeMeanCov(d)
    result = 0.0
    for i=1:p
        result += sigma_weights[i]*g(m+sqrt(2*v)*sigma_points[i])/sqrt(pi)
    end
    return result
end

function laplaceApproximation(dist::ProbabilityDistribution{Univariate,ExponentialLinearQuadratic},x_0,dim::Int64)
    a = dist.params[:a]
    b = dist.params[:b]
    c = dist.params[:c]
    d = dist.params[:d]
    g(x) = -0.5*(a*x+b*exp(c*x+d*x^2/2))
    mean = 0.0
    var = 0.0
    for i=1:dim
        grad = ForwardDiff.derivative(g,x_0)
        hessian = ForwardDiff.derivative(x -> ForwardDiff.derivative(g,x),x_0)
        mean = x_0 - inv(hessian)*grad
        var = -0.5*inv(hessian)
        x_0 = mean
    end
    return ProbabilityDistribution(GaussianMeanVariance,m=mean,v=var)
end
