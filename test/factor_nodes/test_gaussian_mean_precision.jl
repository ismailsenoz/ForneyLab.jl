module GaussianMeanPrecisionTest

using Base.Test
using ForneyLab
import ForneyLab: outboundType, isApplicable
import ForneyLab: VBGaussianMeanPrecisionOut, VBGaussianMeanPrecisionM, VBGaussianMeanPrecisionW


#-------------
# Update rules
#-------------

@testset "VBGaussianMeanPrecisionM" begin
    @test VBGaussianMeanPrecisionM <: VariationalRule{GaussianMeanPrecision}
    @test outboundType(VBGaussianMeanPrecisionM) == Message{Gaussian}
    @test isApplicable(VBGaussianMeanPrecisionM, [ProbabilityDistribution, Void, ProbabilityDistribution]) 
    @test !isApplicable(VBGaussianMeanPrecisionM, [ProbabilityDistribution, ProbabilityDistribution, Void]) 

    @test ruleVBGaussianMeanPrecisionM(ProbabilityDistribution(Univariate, Gaussian, m=3.0, v=4.0), nothing, ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0)) == Message(Univariate, Gaussian, m=3.0, w=0.5)
    @test ruleVBGaussianMeanPrecisionM(ProbabilityDistribution(Multivariate, Gaussian, m=[3.0], v=[4.0].'), nothing, ProbabilityDistribution(MatrixVariate, Wishart, v=[0.25].', nu=2.0)) == Message(Multivariate, Gaussian, m=[3.0], w=[0.5].')
end

@testset "VBGaussianMeanPrecisionW" begin
    @test VBGaussianMeanPrecisionW <: VariationalRule{GaussianMeanPrecision}
    @test outboundType(VBGaussianMeanPrecisionW) == Message{Union{Gamma, Wishart}}
    @test isApplicable(VBGaussianMeanPrecisionW, [ProbabilityDistribution, ProbabilityDistribution, Void]) 

    @test ruleVBGaussianMeanPrecisionW(ProbabilityDistribution(Univariate, Gaussian, m=3.0, v=4.0), ProbabilityDistribution(Univariate, Gaussian, m=1.0, v=2.0), nothing) == Message(Univariate, Gamma, a=1.5, b=0.5*(2.0 + 4.0 + (3.0 - 1.0)^2))
    @test ruleVBGaussianMeanPrecisionW(ProbabilityDistribution(Multivariate, Gaussian, m=[3.0], v=[4.0].'), ProbabilityDistribution(Multivariate, Gaussian, m=[1.0], v=[2.0].'), nothing) == Message(MatrixVariate, Wishart, v=[1.0/(2.0 + 4.0 + (3.0 - 1.0)^2)].', nu=3.0)
end

@testset "VBGaussianMeanPrecisionOut" begin
    @test VBGaussianMeanPrecisionOut <: VariationalRule{GaussianMeanPrecision}
    @test outboundType(VBGaussianMeanPrecisionOut) == Message{Gaussian}
    @test isApplicable(VBGaussianMeanPrecisionOut, [Void, ProbabilityDistribution, ProbabilityDistribution]) 

    @test ruleVBGaussianMeanPrecisionOut(nothing, ProbabilityDistribution(Univariate, Gaussian, m=3.0, v=4.0), ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0)) == Message(Univariate, Gaussian, m=3.0, w=0.5)
    @test ruleVBGaussianMeanPrecisionOut(nothing, ProbabilityDistribution(Multivariate, Gaussian, m=[3.0], v=[4.0].'), ProbabilityDistribution(MatrixVariate, Wishart, v=[0.25].', nu=2.0)) == Message(Multivariate, Gaussian, m=[3.0], w=[0.5].')
end

@testset "averageEnergy and differentialEntropy" begin
    @test differentialEntropy(ProbabilityDistribution(Univariate, Gaussian, m=0.0, w=2.0)) == averageEnergy(GaussianMeanPrecision, ProbabilityDistribution(Univariate, Gaussian, m=0.0, w=2.0), ProbabilityDistribution(Univariate, PointMass, m=0.0), ProbabilityDistribution(Univariate, PointMass, m=2.0))
    @test differentialEntropy(ProbabilityDistribution(Univariate, Gaussian, m=0.0, w=2.0)) == differentialEntropy(ProbabilityDistribution(Multivariate, Gaussian, m=[0.0], w=[2.0].'))
    @test averageEnergy(GaussianMeanPrecision, ProbabilityDistribution(Univariate, Gaussian, m=0.0, w=2.0), ProbabilityDistribution(Univariate, PointMass, m=0.0), ProbabilityDistribution(Univariate, PointMass, m=2.0)) == averageEnergy(GaussianMeanPrecision, ProbabilityDistribution(Multivariate, Gaussian, m=[0.0], w=[2.0].'), ProbabilityDistribution(Multivariate, PointMass, m=[0.0]), ProbabilityDistribution(MatrixVariate, PointMass, m=[2.0].'))
end

end #module