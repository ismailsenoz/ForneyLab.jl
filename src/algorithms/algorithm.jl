export Algorithm, currentAlgorithm

"""
An `Algorithm` holds a collection of (non-overlapping) recognition factors that
specify the recognition factorization over a factor graph.
"""
mutable struct Algorithm
    id::Symbol
    
    graph::FactorGraph
    recognition_factors::Dict{Symbol, RecognitionFactor}

    # Bookkeeping for faster lookup during scheduling
    edge_to_recognition_factor::Dict{Edge, RecognitionFactor}
    node_edge_to_cluster::Dict{Tuple{FactorNode, Edge}, Cluster}

    # Bookkeeping for faster lookup during assembly
    interface_to_schedule_entry::Dict{Interface, ScheduleEntry}
    target_to_marginal_entry::Dict{Union{Variable, Cluster}, MarginalEntry}

    # Fields for free energy algorithm assembly
    average_energies::Vector{Dict{Symbol, Any}}
    entropies::Vector{Dict{Symbol, Any}}
end

"""
Return currently active `Algorithm`.
Create one if there is none.
"""
function currentAlgorithm()
    try
        return current_algorithm
    catch
        return Algorithm()
    end
end

setCurrentAlgorithm(rf::Algorithm) = global current_algorithm = rf

Algorithm(id=Symbol("")) = setCurrentAlgorithm(
    Algorithm(
        id,
        currentGraph(),
        Dict{Symbol, RecognitionFactor}(),
        Dict{Edge, RecognitionFactor}(),
        Dict{Tuple{FactorNode, Edge}, Symbol}(),
        Dict{Interface, ScheduleEntry}(),
        Dict{Union{Variable, Cluster}, MarginalEntry}(),
        Dict{Symbol, Any}[],
        Dict{Symbol, Any}[]))

"""
Construct a `Algorithm` consisting of one
`RecognitionFactor` for each argument
"""
function Algorithm(args::Vararg{Union{T, Set{T}, Vector{T}} where T<:Variable}; ids=Symbol[], id=Symbol(""))
    rf = Algorithm(id)
    isempty(ids) || (length(ids) == length(args)) || error("Length of ids must match length of recognition factor arguments")
    for (i, arg) in enumerate(args)
        if isempty(ids)
            RecognitionFactor(arg, id=generateId(RecognitionFactor))
        else        
            RecognitionFactor(arg, id=ids[i])
        end
    end
    return rf
end

function interfaceToScheduleEntry(algo::Algorithm)
    mapping = Dict{Interface, ScheduleEntry}()
    for (id, rf) in algo.recognition_factors
        rf_mapping = interfaceToScheduleEntry(rf.schedule)
        merge!(mapping, rf_mapping)
    end

    return mapping
end

function targetToMarginalEntry(algo::Algorithm)
    mapping = Dict{Union{Cluster, Variable}, MarginalEntry}()
    for (id, rf) in algo.recognition_factors
        rf_mapping = targetToMarginalEntry(rf.marginal_table)
        merge!(mapping, rf_mapping)
    end

    return mapping    
end