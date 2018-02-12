module RecognitionFactorizationTest

using Base.Test
using ForneyLab

import ForneyLab: nodesConnectedToExternalEdges, Cluster

@testset "RecognitionFactorization" begin
    rf = RecognitionFactorization()
    @test rf.recognition_factors == Dict{Symbol, RecognitionFactor}()
    @test rf.edge_to_recognition_factor == Dict{Edge, RecognitionFactor}()
    @test currentRecognitionFactorization() === rf
end

@testset "RecognitionFactor" begin
    g = FactorGraph()
    @RV m ~ GaussianMeanVariance(constant(0.0), constant(1.0))
    @RV w ~ Gamma(constant(1.0), constant(1.0))
    y = Variable[]
    for i = 1:3
        @RV y_i ~ GaussianMeanPrecision(m, w)
        placeholder(y_i, :y, index=i)
        push!(y, y_i)
    end

    rf = RecognitionFactorization()
    q_m = RecognitionFactor(m)
    @test q_m.id == :recognitionfactor_1
    @test q_m.variables == Set([m])
    @test q_m.clusters == Set{Cluster}()
    @test q_m.internal_edges == edges(m)
    @test rf.recognition_factors[:recognitionfactor_1] === q_m

    q_w = RecognitionFactor(w)
    @test q_w.id == :recognitionfactor_2
    @test q_w.variables == Set([w])
    @test q_w.clusters == Set{Cluster}()
    @test q_w.internal_edges == edges(w)
    @test rf.recognition_factors[:recognitionfactor_2] === q_w

    # Joint factorizations
    q_m_w = RecognitionFactor([m, w])
    @test q_m_w.id == :recognitionfactor_3
    @test q_m_w.variables == Set([m, w])
    @test length(q_m_w.clusters) == 3 
    @test q_m_w.internal_edges == edges(Set([m, w]))
    @test rf.recognition_factors[:recognitionfactor_3] === q_m_w

    q_y = RecognitionFactor(y)
    @test q_y.id == :recognitionfactor_4
    @test q_y.variables == Set(y)
    @test q_y.clusters == Set{Cluster}()
    @test q_y.internal_edges == edges(Set(y))
    @test rf.recognition_factors[:recognitionfactor_4] === q_y
end

@testset "Cluster" begin
    g = FactorGraph()

    m = Variable(id=:m)
    v = Variable(id=:v)
    y = Variable(id=:y)
    nd = GaussianMeanVariance(y, m, v)
    em = nd.i[:m].edge
    ev = nd.i[:v].edge

    cluster = Cluster(nd, [em, ev])

    @test cluster.id == :m_v
    @test cluster.node == nd
    @test cluster.edges[1] == em
    @test cluster.edges[2] == ev
end

end # module