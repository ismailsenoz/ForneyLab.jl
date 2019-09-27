export
ruleSPEqualityGaussian,
ruleSPEqualityGammaWishart,
ruleSPEqualityBernoulli,
ruleSPEqualityBeta,
ruleSPEqualityCategorical,
ruleSPEqualityDirichlet,
ruleSPEqualityPointMass,
ruleSPEqualityILE,
ruleSPEqualityGaussianILE,
ruleSPEqualityGaussianLDT

ruleSPEqualityGaussianLDT(msg_1::Message{F1,Multivariate},msg_2::Message{F2},msg_3::Nothing) where {F1<:Gaussian, F2<:LogDetTrace} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianLDT(msg_1::Message{F2},msg_2::Message{F1,Multivariate},msg_3::Nothing) where {F1<:Gaussian, F2<:LogDetTrace} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianLDT(msg_1::Message{F1,Multivariate},msg_2::Nothing,msg_3::Message{F2}) where {F1<:Gaussian, F2<:LogDetTrace} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianLDT(msg_1::Message{F2},msg_2::Nothing,msg_3::Message{F1,Multivariate}) where {F1<:Gaussian, F2<:LogDetTrace} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianLDT(msg_1::Nothing,msg_2::Message{F1,Multivariate},msg_3::Message{F2}) where {F1<:Gaussian, F2<:LogDetTrace} = Message(prod!(msg_2.dist,msg_3.dist))
ruleSPEqualityGaussianLDT(msg_1::Nothing,msg_2::Message{F2},msg_3::Message{F1,Multivariate}) where {F1<:Gaussian, F2<:LogDetTrace} = Message(prod!(msg_2.dist,msg_3.dist))

ruleSPEqualityGaussian(msg_1::Message{F1}, msg_2::Message{F2}, msg_3::Nothing) where {F1<:Gaussian, F2<:Gaussian} = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityGaussian(msg_1::Message{F1}, msg_2::Nothing, msg_3::Message{F2}) where {F1<:Gaussian, F2<:Gaussian}= Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityGaussian(msg_1::Nothing, msg_2::Message{F1}, msg_3::Message{F2}) where {F1<:Gaussian, F2<:Gaussian} = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityGaussianILE(msg_1::Message{F}, msg_2::Message{InverseLinearExponential}, msg_3::Nothing) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityGaussianILE(msg_1::Message{InverseLinearExponential}, msg_2::Message{F}, msg_3::Nothing) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityGaussianILE(msg_1::Message{F}, msg_2::Nothing, msg_3::Message{InverseLinearExponential}) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityGaussianILE(msg_1::Message{InverseLinearExponential}, msg_2::Nothing, msg_3::Message{F}) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityGaussianILE(msg_1::Nothing, msg_2::Message{F}, msg_3::Message{InverseLinearExponential}) where {F<:Gaussian} = Message(prod!(msg_2.dist, msg_3.dist))
ruleSPEqualityGaussianILE(msg_1::Nothing, msg_2::Message{InverseLinearExponential}, msg_3::Message{F}) where {F<:Gaussian} = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityILE(msg_1::Message{InverseLinearExponential}, msg_2::Message{InverseLinearExponential}, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityILE(msg_1::Message{InverseLinearExponential}, msg_2::Nothing, msg_3::Message{InverseLinearExponential}) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityILE(msg_1::Nothing, msg_2::Message{InverseLinearExponential}, msg_3::Message{InverseLinearExponential}) = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityLLE(msg_1::Message{F}, msg_2::Message{LogLinearExponential}, msg_3::Nothing) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityLLE(msg_1::Message{LogLinearExponential}, msg_2::Message{F}, msg_3::Nothing) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityLLE(msg_1::Message{F}, msg_2::Nothing, msg_3::Message{LogLinearExponential}) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityLLE(msg_1::Message{LogLinearExponential}, msg_2::Nothing, msg_3::Message{F}) where {F<:Gaussian} = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityLLE(msg_1::Nothing, msg_2::Message{F}, msg_3::Message{LogLinearExponential}) where {F<:Gaussian} = Message(prod!(msg_2.dist, msg_3.dist))
ruleSPEqualityLLE(msg_1::Nothing, msg_2::Message{LogLinearExponential}, msg_3::Message{F}) where {F<:Gaussian} = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityGammaWishart(msg_1::Message{F}, msg_2::Message{F}, msg_3::Nothing) where F<:Union{Gamma, Wishart} = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityGammaWishart(msg_1::Message{F}, msg_2::Nothing, msg_3::Message{F}) where F<:Union{Gamma, Wishart}= Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityGammaWishart(msg_1::Nothing, msg_2::Message{F}, msg_3::Message{F}) where F<:Union{Gamma, Wishart} = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityBernoulli(msg_1::Message{Bernoulli}, msg_2::Message{Bernoulli}, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityBernoulli(msg_1::Message{Bernoulli}, msg_2::Nothing, msg_3::Message{Bernoulli}) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityBernoulli(msg_1::Nothing, msg_2::Message{Bernoulli}, msg_3::Message{Bernoulli}) = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityBeta(msg_1::Message{Beta}, msg_2::Message{Beta}, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityBeta(msg_1::Message{Beta}, msg_2::Nothing, msg_3::Message{Beta}) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityBeta(msg_1::Nothing, msg_2::Message{Beta}, msg_3::Message{Beta}) = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityCategorical(msg_1::Message{Categorical}, msg_2::Message{Categorical}, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityCategorical(msg_1::Message{Categorical}, msg_2::Nothing, msg_3::Message{Categorical}) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityCategorical(msg_1::Nothing, msg_2::Message{Categorical}, msg_3::Message{Categorical}) = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityDirichlet(msg_1::Message{Dirichlet}, msg_2::Message{Dirichlet}, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityDirichlet(msg_1::Message{Dirichlet}, msg_2::Nothing, msg_3::Message{Dirichlet}) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityDirichlet(msg_1::Nothing, msg_2::Message{Dirichlet}, msg_3::Message{Dirichlet}) = Message(prod!(msg_2.dist, msg_3.dist))

ruleSPEqualityPointMass(msg_1::Message, msg_2::Message, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityPointMass(msg_1::Message, msg_2::Nothing, msg_3::Message) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityPointMass(msg_1::Nothing, msg_2::Message, msg_3::Message) = Message(prod!(msg_2.dist, msg_3.dist))
