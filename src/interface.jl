export Interface
export clearMessage!, setMessage!, message, handle

"""
An Interface belongs to a node and is used to send/receive messages.
An Interface can be seen as a half-edge; it has exactly one partner interface, with wich it forms an edge.
A message from node a to node b is stored at the Interface of node a that connects to an Interface of node b.
"""
type Interface
    node::Node
    edge::Union{AbstractEdge, Void}
    partner::Union{Interface, Void}
    message::Union{Message, Void}
end
Interface(node::Node) = Interface(node, nothing, nothing, nothing)

function show(io::IO, interface::Interface)
    iface_handle = handle(interface)
    (iface_handle == "") || (iface_handle = "($(iface_handle))")
    println(io, "Interface $(findfirst(interface.node.interfaces, interface)) $(iface_handle) of $(typeof(interface.node)) $(interface.node.id)")
end

Base.deepcopy(::Interface) = error("deepcopy(::Interface) is not possible. An Interface should only be created by a Node constructor.")

function setMessage!(interface::Interface, message::Message)
    interface.message = deepcopy(message)
end

clearMessage!(interface::Interface) = (interface.message = nothing)

function handle(interface::Interface)
    # Return named interface handle
    if isdefined(interface.node, :i)
        for h in keys(interface.node.i)
            if (typeof(h)==Symbol || typeof(h)==Int) && is(interface.node.i[h], interface)
                return string(h)
            end
        end
    end

    return ""
end

function ensureMessage!{T<:ProbabilityDistribution}(interface::Interface, payload_type::Type{T})
    # Ensure that interface carries a Message{payload_type}, used for in place updates
    if interface.message == nothing || typeof(interface.message.payload) != payload_type
        if payload_type <: DeltaDistribution{Float64}
            interface.message = Message(DeltaDistribution())
        elseif payload_type <: DeltaDistribution{Bool}
            interface.message = Message(DeltaDistribution(false))
        elseif payload_type <: MvDeltaDistribution
            dims = payload_type.parameters[end]
            interface.message = Message(MvDeltaDistribution(zeros(dims)))
        elseif payload_type <: MatrixDeltaDistribution
            dims_n = payload_type.parameters[end-1]
            dims_m = payload_type.parameters[end]
            interface.message = Message(MatrixDeltaDistribution(zeros(dims_n, dims_m)))
        else
            interface.message = Message(vague(payload_type))
        end
    end

    return interface.message
end
