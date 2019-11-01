module GaussianControlledVarianceTest

using Test
using ForneyLab
import ForneyLab: outboundType, isApplicable, isProper, unsafeMean, unsafeMode, unsafeVar, unsafeCov, unsafeMeanCov, unsafePrecision, unsafeWeightedMean, unsafeWeightedMeanPrecision
import ForneyLab: SVBGaussianControlledVarianceOutNGDDD, SVBGaussianControlledVarianceXGNDDD,SVBGaussianControlledVarianceZDNDD,SVBGaussianControlledVarianceΚDDND,SVBGaussianControlledVarianceΩDDDN,MGaussianControlledVarianceGGDDD



@testset "SVBGaussianControlledVarianceOutNGDDD" begin
    @test SVBGaussianControlledVarianceOutNGDDD <: StructuredVariationalRule{GaussianControlledVariance}
    @test outboundType(SVBGaussianControlledVarianceOutNGDDD) == Message{GaussianMeanVariance}
    @test isApplicable(SVBGaussianControlledVarianceOutNGDDD, [Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])
    @test !isApplicable(SVBGaussianControlledVarianceOutNGDDD, [Message{Gaussian}, Nothing, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])

    @test ruleSVBGaussianControlledVarianceOutNGDDD(nothing, Message(Univariate, GaussianMeanVariance, m=0.0,v=1.0), ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0)) == Message(Univariate, GaussianMeanVariance, m=0.0, v=1.3678794411714423)
end

@testset "SVBGaussianControlledVarianceXGNDDD" begin
    @test SVBGaussianControlledVarianceXGNDDD <: StructuredVariationalRule{GaussianControlledVariance}
    @test outboundType(SVBGaussianControlledVarianceXGNDDD) == Message{GaussianMeanVariance}
    @test !isApplicable(SVBGaussianControlledVarianceXGNDDD, [Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])
    @test isApplicable(SVBGaussianControlledVarianceXGNDDD, [Message{Gaussian},Nothing, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])

    @test ruleSVBGaussianControlledVarianceXGNDDD(Message(Univariate,GaussianMeanVariance, m=0.0,v=1.0), nothing, ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0)) == Message(Univariate, GaussianMeanVariance, m=0.0, v=1.3678794411714423)
end

@testset "SVBGaussianControlledVarianceZDNDD" begin
    @test SVBGaussianControlledVarianceZDNDD <: StructuredVariationalRule{GaussianControlledVariance}
    @test outboundType(SVBGaussianControlledVarianceZDNDD) == Message{ExponentialLinearQuadratic}
    @test !isApplicable(SVBGaussianControlledVarianceZDNDD, [Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])
    @test isApplicable(SVBGaussianControlledVarianceZDNDD, [ProbabilityDistribution, Nothing,ProbabilityDistribution,ProbabilityDistribution])

    @test ruleSVBGaussianControlledVarianceZDNDD(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)), nothing, ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0)) == Message(Univariate, ExponentialLinearQuadratic,a=0.00, b=3.2974425414002564, c=-0.00, d=1.00)
end

@testset "SVBGaussianControlledVarianceΚDDND" begin
    @test SVBGaussianControlledVarianceΚDDND <: StructuredVariationalRule{GaussianControlledVariance}
    @test outboundType(SVBGaussianControlledVarianceΚDDND) == Message{ExponentialLinearQuadratic}
    @test !isApplicable(SVBGaussianControlledVarianceΚDDND, [Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])
    @test isApplicable(SVBGaussianControlledVarianceΚDDND, [ProbabilityDistribution,ProbabilityDistribution,Nothing,ProbabilityDistribution])

    @test ruleSVBGaussianControlledVarianceΚDDND(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),nothing,ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0)) == Message(Univariate, ExponentialLinearQuadratic,a=0.00, b=3.2974425414002564, c=-0.00, d=1.00)
end

@testset "SVBGaussianControlledVarianceΩDDDN" begin
    @test SVBGaussianControlledVarianceΩDDDN <: StructuredVariationalRule{GaussianControlledVariance}
    @test outboundType(SVBGaussianControlledVarianceΩDDDN) == Message{ExponentialLinearQuadratic}
    @test !isApplicable(SVBGaussianControlledVarianceΩDDDN, [Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])
    @test isApplicable(SVBGaussianControlledVarianceΩDDDN, [ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,Nothing])

    @test ruleSVBGaussianControlledVarianceΩDDDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),nothing) == Message(Univariate, ExponentialLinearQuadratic,a=1.00, b=3.2974425414002564, c=-1.00, d=0.00)
end


@testset "MGaussianControlledVarianceGGDDD" begin
    @test MGaussianControlledVarianceGGDDD <: MarginalRule{GaussianControlledVariance}
    @test isApplicable(MGaussianControlledVarianceGGDDD, [Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution])

    @test ruleMGaussianControlledVarianceGGDDD(Message(Univariate, GaussianMeanVariance, m=0.0, v=1.0), Message(Univariate, GaussianMeanVariance, m=0.0, v=1.0), ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0),ProbabilityDistribution(Univariate,GaussianMeanVariance,m=0.0,v=1.0)) == ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=[0.00, 0.00], w=[3.7182818284590455 -2.7182818284590455;-2.7182818284590455 3.7182818284590455])
end

@testset "averageEnergy" begin
    @test averageEnergy(GaussianControlledVariance, ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(2),v=diageye(2)), ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0), ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0),ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) == -5.400627603542738
end

end #module
