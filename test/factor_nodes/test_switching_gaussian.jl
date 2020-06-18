module SwitchingGaussianTest


using Test
using ForneyLab
using ForneyLab: outboundType, isApplicable, isProper, unsafeMean, unsafeMode, unsafeVar, unsafeCov, unsafeMeanCov, unsafePrecision, unsafeWeightedMean, unsafeWeightedMeanPrecision
using ForneyLab: SVBSwitchingGaussianOutNGD, SVBSwitchingGaussianMGND, SVBSwitchingGaussianSDN, MSwitchingGaussianGGD

@testset "SVBSwitchingGaussianOutNGD" begin
    @test SVBSwitchingGaussianOutNGD <: StructuredVariationalRule{SwitchingGaussian}
    @test outboundType(SVBSwitchingGaussianOutNGD) == Message{GaussianMeanVariance}
    @test isApplicable(SVBSwitchingGaussianOutNGD, [Nothing, Message{Gaussian},ProbabilityDistribution])
    #
    cat = ProbabilityDistribution(Univariate,Categorical,p=[1/2 1/2])
    dist_A = [randn(2,2) for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    @test ruleSVBSwitchingGaussianOutNGD(nothing, Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)), cat,dist_A,dist_Q) == Message(Multivariate, GaussianMeanVariance, m=zeros(2), v=2*diageye(2))
    @test ruleSVBSwitchingGaussianMGND(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)), nothing,cat,dist_A,dist_Q) == Message(Multivariate, GaussianMeanVariance, m=zeros(2), v=2*diageye(2))
end

@testset "SVBSwitchingGaussianSDN" begin
    @test SVBSwitchingGaussianSDN <: StructuredVariationalRule{SwitchingGaussian}
    @test outboundType(SVBSwitchingGaussianSDN) == Message{Categorical}
    @test isApplicable(SVBSwitchingGaussianSDN, [ProbabilityDistribution,Nothing])

    dist_A = [randn(2,2) for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    p = ruleSVBSwitchingGaussianSDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(4),v=diageye(4)),nothing,dist_A,dist_Q).dist.params[:p]
    @test ruleSVBSwitchingGaussianSDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(4),v=diageye(4)),nothing,dist_A,dist_Q) == Message(Categorical,p=p)
end

@testset "MSwitchingGaussianGGD" begin
    @test MSwitchingGaussianGGD <: MarginalRule{SwitchingGaussian}
    @test isApplicable(MSwitchingGaussianGGD, [Message{Gaussian},Message{Gaussian},ProbabilityDistribution])
    cat = ProbabilityDistribution(Univariate,Categorical,p=[1/2 1/2])
    dist_A = [randn(2,2) for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    xi, W = unsafeWeightedMeanPrecision(ruleMSwitchingGaussianGGD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),cat,dist_A,dist_Q))
    @test ruleMSwitchingGaussianGGD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),cat,dist_A,dist_Q) == ProbabilityDistribution(Multivariate,GaussianWeightedMeanPrecision,xi=xi,w=W)
end

#
@testset "averageEnergy" begin
    dist_A = [[1.0 0.0; 0.0 1.0] for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    @test averageEnergy(SwitchingGaussian, ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(4), v=diageye(4)), ProbabilityDistribution(Categorical, p=[1/2, 1/2]), dist_A,dist_Q) == 3.8378770664093453
end


end #module
