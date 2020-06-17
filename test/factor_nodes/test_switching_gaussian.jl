module SwitchingGaussianTest


using Test
using ForneyLab
using ForneyLab: outboundType, isApplicable, isProper, unsafeMean, unsafeMode, unsafeVar, unsafeCov, unsafeMeanCov, unsafePrecision, unsafeWeightedMean, unsafeWeightedMeanPrecision
using ForneyLab: SVBSwitchingGaussianOutNGDDD, SVBSwitchingGaussianMGNDDD, SVBSwitchingGaussianSDNDD, MSwitchingGaussianGGDDD

@testset "SVBSwitchingGaussianOutNGDDD" begin
    @test SVBSwitchingGaussianOutNGDDD <: StructuredVariationalRule{SwitchingGaussian}
    @test outboundType(SVBSwitchingGaussianOutNGDDD) == Message{GaussianMeanVariance}
    @test isApplicable(SVBSwitchingGaussianOutNGDDD, [Nothing, Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution, ProbabilityDistribution])

    cat = ProbabilityDistribution(Univariate,Categorical,p=[1/2 1/2])
    list_A = [randn(2,2) for i=1:2]
    list_Q = [diageye(2) for i=1:2]
    dist_A = ProbabilityDistribution(MatrixVariate,SampleList,s=list_A,w=ones(2)/2)
    dist_Q = ProbabilityDistribution(MatrixVariate,SampleList,s=list_Q,w=ones(2)/2)
    @test ruleSVBSwitchingGaussianOutNGDDD(nothing, Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)), cat,dist_A,dist_Q) == Message(Multivariate, GaussianMeanVariance, m=zeros(2), v=2*diageye(2))
    @test ruleSVBSwitchingGaussianMGNDDD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)), nothing,cat,dist_A,dist_Q) == Message(Multivariate, GaussianMeanVariance, m=zeros(2), v=2*diageye(2))
end

@testset "SVBSwitchingGaussianSDNDD" begin
    @test SVBSwitchingGaussianSDNDD <: StructuredVariationalRule{SwitchingGaussian}
    @test outboundType(SVBSwitchingGaussianSDNDD) == Message{Categorical}
    @test isApplicable(SVBSwitchingGaussianSDNDD, [ProbabilityDistribution,Nothing,ProbabilityDistribution, ProbabilityDistribution])

    list_A = [randn(2,2) for i=1:2]
    list_Q = [diageye(2) for i=1:2]
    dist_A = ProbabilityDistribution(MatrixVariate,SampleList,s=list_A,w=ones(2)/2)
    dist_Q = ProbabilityDistribution(MatrixVariate,SampleList,s=list_Q,w=ones(2)/2)
    p = ruleSVBSwitchingGaussianSDNDD(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(4),v=diageye(4)),nothing,dist_A,dist_Q).dist.params[:p]
    @test ruleSVBSwitchingGaussianSDNDD(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(4),v=diageye(4)),nothing,dist_A,dist_Q) == Message(Categorical,p=p)
end

@testset "MSwitchingGaussianGGDDD" begin
    @test MSwitchingGaussianGGDDD <: MarginalRule{SwitchingGaussian}
    @test isApplicable(MSwitchingGaussianGGDDD, [Message{Gaussian},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution, ProbabilityDistribution])
    cat = ProbabilityDistribution(Univariate,Categorical,p=[1/2 1/2])
    list_A = [randn(2,2) for i=1:2]
    list_Q = [diageye(2) for i=1:2]
    dist_A = ProbabilityDistribution(MatrixVariate,SampleList,s=list_A,w=ones(2)/2)
    dist_Q = ProbabilityDistribution(MatrixVariate,SampleList,s=list_Q,w=ones(2)/2)
    xi, W = unsafeWeightedMeanPrecision(ruleMSwitchingGaussianGGDDD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),cat,dist_A,dist_Q))
    @test ruleMSwitchingGaussianGGDDD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),cat,dist_A,dist_Q) == ProbabilityDistribution(Multivariate,GaussianWeightedMeanPrecision,xi=xi,w=W)
end


@testset "averageEnergy" begin
    list_A = [diageye(2) for i=1:2]
    list_Q = [diageye(2) for i=1:2]
    dist_A = ProbabilityDistribution(MatrixVariate,SampleList,s=list_A,w=ones(2)/2)
    dist_Q = ProbabilityDistribution(MatrixVariate,SampleList,s=list_Q,w=ones(2)/2)
    @test averageEnergy(SwitchingGaussian, ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(4), v=diageye(4)), ProbabilityDistribution(Categorical, p=[1/2, 1/2]), dist_A,dist_Q) == 3.8378770664093453
end


end #module
