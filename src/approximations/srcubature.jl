
import LinearAlgebra: mul!

struct SphericalRadialCubature
    ndims :: Int
end

srcubature(ndims::Int) = SphericalRadialCubature(ndims)

function getweights(cubature::SphericalRadialCubature)
    d = cubature.ndims
    return Base.Generator(1:2d + 1) do i
        return i === (2d + 1) ? 1.0 / (d + 1) : 1.0 / (2.0(d + 1))
        # return 1.0 / (2.0 * d + 1)
    end
end

function getpoints(cubature::SphericalRadialCubature, m, P)
    d = cubature.ndims

    if isa(P, Diagonal)
        L = sqrt(P) # Matrix square root
    else
        L = sqrt(Hermitian(P))
    end

    tmpbuffer = zeros(d)
    sigma_points = Base.Generator(1:2d + 1) do i
        if i === (2d + 1)
            fill!(tmpbuffer, 0.0)
        else
            tmpbuffer[rem((i - 1), d) + 1] = sqrt(d + 1) * (-1)^(div(i - 1, d))
            if i !== 1
                tmpbuffer[rem((i - 2), d) + 1] = 0.0
            end
        end
        return tmpbuffer
    end

    tbuffer = similar(m)
    return Base.Generator(sigma_points) do point
        copyto!(tbuffer, m)
        return mul!(tbuffer, L, point, 1.0, 1.0) # point = m + 1.0 * L * point
    end
end

function approximate_meancov(cubature::SphericalRadialCubature, g, distribution)
    ndims = cubature.ndims

    c    = approximate_kernel_expectation(cubature, g, distribution)
    mean = approximate_kernel_expectation(cubature, (s) -> g(s) * s / c, distribution)
    cov  = approximate_kernel_expectation(cubature, (s) -> g(s) * (s - mean) * (s - mean)' / c, distribution)

    # c    = unscentedStatistics(cubature, g, distribution)
    # mean = unscentedStatistics(cubature, (s) -> g(s) * s / c, distribution)
    # cov  = unscentedStatistics(cubature, (s) -> g(s) * (s - mean) * (s - mean)' / c, distribution)

    # @show c

    return mean, cov
end

function unscentedStatistics(::SphericalRadialCubature, g::Function, distribution; alpha=1.0, beta=0.0, kappa=1.0)
    m, V = ForneyLab.unsafeMeanCov(distribution)
    (sigma_points, weights_m, weights_c) = ForneyLab.sigmaPointsAndWeights(m, V; alpha=alpha, beta=beta, kappa=kappa)
    d = length(m)

    g_sigma = g.(sigma_points)
    m_tilde = sum([weights_m[k+1]*g_sigma[k+1] for k=0:2*d])

    return m_tilde
end

function approximate_kernel_expectation(cubature::SphericalRadialCubature, g, distribution)
    m, V = ForneyLab.unsafeMeanCov(distribution)

    weights = getweights(cubature)
    points  = getpoints(cubature, m, V)

    gs = Base.Generator(points) do point
        return g(point)
    end

    return mapreduce(t -> t[1] * t[2], +, zip(weights, gs))
end
