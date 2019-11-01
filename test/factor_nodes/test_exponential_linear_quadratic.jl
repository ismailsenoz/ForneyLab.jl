module ExponentialLinearQuadraticTest

using Test
using ForneyLab
import ForneyLab:ExponentialLinearQuadratic

@testset "prod!" begin
    @test ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0) * ProbabilityDistribution(Univariate, ExponentialLinearQuadratic, a=1.0,b=1.0,c=1.0,d=1.0) == ProbabilityDistribution(Univariate, GaussianMeanVariance, m=-0.4354337072954266, v=0.35209695558257903)
end

end #module
