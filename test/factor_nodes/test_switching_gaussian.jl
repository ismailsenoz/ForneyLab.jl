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

    dist_A = [randn(1) for i=1:2]
    dist_Q = [[1.0] for i=1:2]
    @test ruleSVBSwitchingGaussianOutNGD(nothing, Message(Univariate,GaussianMeanVariance,m=0.0,v=1.0), cat,dist_A,dist_Q) == Message(Univariate, GaussianMeanVariance, m=0.0, v=2.00)
    @test ruleSVBSwitchingGaussianMGND(Message(Univariate,GaussianMeanVariance,m=0.0,v=1.0), nothing,cat,dist_A,dist_Q) == Message(Univariate, GaussianMeanVariance, m=0.0, v=2.00)

end

@testset "SVBSwitchingGaussianSDN" begin
    @test SVBSwitchingGaussianSDN <: StructuredVariationalRule{SwitchingGaussian}
    @test outboundType(SVBSwitchingGaussianSDN) == Message{Categorical}
    @test isApplicable(SVBSwitchingGaussianSDN, [ProbabilityDistribution,Nothing])

    dist_A = [randn(2,2) for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    p = ruleSVBSwitchingGaussianSDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(4),v=diageye(4)),nothing,dist_A,dist_Q).dist.params[:p]
    @test ruleSVBSwitchingGaussianSDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(4),v=diageye(4)),nothing,dist_A,dist_Q) == Message(Categorical,p=p)

    dist_A = [randn(1) for i=1:2]
    dist_Q = [[1.0] for i=1:2]
    p = ruleSVBSwitchingGaussianSDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),nothing,dist_A,dist_Q).dist.params[:p]
    @test ruleSVBSwitchingGaussianSDN(ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),nothing,dist_A,dist_Q) == Message(Categorical,p=p)
end

@testset "MSwitchingGaussianGGD" begin
    @test MSwitchingGaussianGGD <: MarginalRule{SwitchingGaussian}
    @test isApplicable(MSwitchingGaussianGGD, [Message{Gaussian},Message{Gaussian},ProbabilityDistribution])
    cat = ProbabilityDistribution(Univariate,Categorical,p=[1/2 1/2])
    dist_A = [randn(2,2) for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    xi, W = unsafeWeightedMeanPrecision(ruleMSwitchingGaussianGGD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),cat,dist_A,dist_Q))
    @test ruleMSwitchingGaussianGGD(Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),Message(Multivariate,GaussianMeanVariance,m=zeros(2),v=diageye(2)),cat,dist_A,dist_Q) == ProbabilityDistribution(Multivariate,GaussianWeightedMeanPrecision,xi=xi,w=W)

    dist_A = [randn(1) for i=1:2]
    dist_Q = [[1.0] for i=1:2]
    xi, W = unsafeWeightedMeanPrecision(ruleMSwitchingGaussianGGD(Message(Univariate,GaussianMeanVariance,m=0.0,v=1.00),Message(Univariate,GaussianMeanVariance,m=0.0,v=1.0),cat,dist_A,dist_Q))
    @test ruleMSwitchingGaussianGGD(Message(Univariate,GaussianMeanVariance,m=0.0,v=1.0),Message(Univariate,GaussianMeanVariance,m=0.0,v=1.0),cat,dist_A,dist_Q) == ProbabilityDistribution(Multivariate,GaussianWeightedMeanPrecision,xi=xi,w=W)
end
#
#
@testset "averageEnergy" begin
    dist_A = [[1.0 0.0; 0.0 1.0] for i=1:2]
    dist_Q = [[1.0 0.0; 0.0 1.0] for i=1:2]
    @test averageEnergy(SwitchingGaussian, ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(4), v=diageye(4)), ProbabilityDistribution(Categorical, p=[1/2, 1/2]), dist_A,dist_Q) == 3.8378770664093453

    dist_A = [[1.0] for i=1:2]
    dist_Q = [[1.0] for i=1:2]
    @test averageEnergy(SwitchingGaussian, ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(2), v=diageye(2)), ProbabilityDistribution(Categorical, p=[1/2, 1/2]), dist_A,dist_Q) == 1.9189385332046727
end

@testset "VMP algorithm generation" begin
    FactorGraph()

    g = FactorGraph()
    A_list = [[1.0],[1.0]]
    Q_list = [[1.0],[0.1]]

    @RV z ~ ForneyLab.Dirichlet([0.5, 0.5])
    @RV s ~ ForneyLab.Categorical(z)
    @RV x_t_min ~ GaussianMeanVariance(placeholder(:mx_t_min),placeholder(:vx_t_min))
    @RV x_t ~ SwitchingGaussian(x_t_min,s,A=A_list,Q=Q_list)
    @RV y ~ GaussianMeanVariance(x_t, 0.1*diageye(1))

    placeholder(y, :y)

    q = PosteriorFactorization([x_t;x_t_min], s, z, ids=[:X, :S, :Z])
    algo = variationalAlgorithm(q, free_energy=true)
    source_code = algorithmSourceCode(algo, free_energy=true)
    @test occursin("ruleSVBSwitchingGaussianOutNGD(nothing, messages[1], marginals[:s], Array{Float64,1}[[1.0], [1.0]], Array{Float64,1}[[1.0], [0.1]])", source_code)
end
end #module
