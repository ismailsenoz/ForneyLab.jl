module GCVTest

using Test
using ForneyLab, KernelFunctions
using ForneyLab: outboundType, isApplicable, isProper, unsafeMean, unsafeMode, unsafeVar, unsafeCov, unsafeMeanCov, unsafePrecision, unsafeWeightedMean, unsafeWeightedMeanPrecision
using LinearAlgebra: det, diag


@testset "averageEnergy" begin
    k1 = SqExponentialKernel()
    g(z) = kernelmatrix(k1, reshape(collect(z),:,1), obsdim=1)+exp.(reshape(collect(z),:,1) + [-2; -1]) .* diageye(2)
    println(averageEnergy(GCV{Cubature}, ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(4), v=diageye(4)), ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(2),v=diageye(2)),g))
end

end
