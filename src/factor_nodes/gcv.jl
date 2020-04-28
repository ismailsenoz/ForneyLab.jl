export GCV, Cubature, Laplace

abstract type ApproximationMethod end
abstract type Cubature <: ApproximationMethod end
abstract type Laplace <: ApproximationMethod end


mutable struct GCV{T<:ApproximationMethod} <: SoftFactor
    id::Symbol
    interfaces::Array{Interface}
    i::Dict{Symbol, Interface}
    g::Function # Vector function that expresses the output vector as a function of the input

    function GCV{Cubature}(out, m, z, g::Function; id=ForneyLab.generateId(GCV{Cubature}))
        @ensureVariables(out, m, z)
        self = new(id, Vector{Interface}(undef, 3), Dict{Symbol,Interface}(), g)
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)

        return self
    end

    function GCV{Laplace}(out, m, z, g::Function; id=ForneyLab.generateId(GCV{Laplace}))
        @ensureVariables(out, m, z)
        self = new(id, Vector{Interface}(undef, 3), Dict{Symbol,Interface}(), g)
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:m] = self.interfaces[2] = associate!(Interface(self), m)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)

        return self
    end
end

function GCV(out, in1, g::Function; id=ForneyLab.generateId(GCV{Cubature}))
    return GCV{Cubature}(out, m, z, g::Function;id=id)
end

slug(::Type{GCV}) = "GCV"
