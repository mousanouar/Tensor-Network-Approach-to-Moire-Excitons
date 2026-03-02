module Hamiltonians


using LinearAlgebra
using ITensors
using ITensorMPS
import TensorCrossInterpolation as TCI
import QuanticsTCI as QTCI
using Quantics

# Load your module (must be in the same directory)

include("kin_builders.jl")
using .kin_builders



########################
# Model registry + builder
########################

# Simple param parser: "k=v,k2=v2" -> Dict{Symbol,Any}
# (bool/int/float detection; else keep as String)
"""
    _parse_param_string(s::AbstractString) -> Dict{Symbol,Any}

Parses a parameter string of the form `"key1=val1, key2=val2, ..."` into a dictionary mapping symbols to values.

# Arguments
- `s::AbstractString`: The input string containing key-value pairs separated by commas, spaces, or tabs. Each key-value pair should be in the form `key=value`.

# Returns
- `Dict{Symbol,Any}`: A dictionary where keys are symbols and values are parsed as `Bool`, `Int`, `Float64`, or left as `String` if they do not match those types.

# Parsing Rules
- Boolean values: `"true"` or `"false"` (case-insensitive) are parsed as `Bool`.
- Integer values: Strings matching an integer pattern are parsed as `Int`.
- Floating-point values: Strings matching a float pattern (including scientific notation) are parsed as `Float64`.
- All other values are kept as `String`.

# Errors
Throws an error if a token does not match the expected `key=value` format.
"""
function _parse_param_string(s::AbstractString)
    d = Dict{Symbol,Any}()
    t = strip(s)
    isempty(t) && return d
    for tok in split(t, [',',' ','\t'])
        isempty(tok) && continue
        kv = split(tok, '=', limit=2)
        length(kv) == 2 || error("Bad model param token: '$tok' (expected key=value)")
        k = Symbol(strip(kv[1])); v = strip(kv[2])
        vl = lowercase(v)
        val::Any =
            vl in ("true","false") ? (vl == "true") :
            occursin(r"^[+-]?\d+$", v) ? parse(Int, v) :
            occursin(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$", v) ? parse(Float64, v) :
            v
        d[k] = val
    end
    return d
end

# Registry maps a model name to:
#   (function symbol, dimension, required positional arg names in order,
#    known keyword defaults, interacting::Bool)
const MODEL_REGISTRY = Dict{String,Tuple{Symbol,Int,Vector{Symbol},NamedTuple,Bool}}(
    "aah"              => (:HAAH,              1, [:V, :phi, :t], (; b=((1 + sqrt(5))/2), tol_quantics=1e-8, maxbonddim_quantics=50), false),
    "ssh"              => (:HSSH,              1, [:t, :d],       (; tol_quantics=1e-8, maxbonddim_quantics=100, nn=1), false),
    "uniform"          => (:HUniform,          1, [:t],           (; v=1e-6, tol_quantics=1e-8, maxbonddim_quantics=100, nn=1), false),
    "uniform2dsquare"  => (:HUniform2Dsquare,  2, [:t],           (; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-10), false),
    "uniform2dhex"     => (:HUniform2Dhex,     2, [:t],           (; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-10), false),
    "uniform2dtri"     => (:HUniform2Dtri,     2, [:t],           (; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-10), false),
    "qc2dsquare"       => (:HQC2Dsquare,       2, [:t],           (; tol_quantics=1e-9, maxbonddim_quantics=250, cutoff=1e-10), false),

    # --- Chern-type Hamiltonians ---
    "chern8"           => (:HChern8,           2, [:V, :t],       (; t2=0.2, tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-8), false),
    "chernhex"         => (:H2DChernhex,       2, [:t],           (; t2=0.2, tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-8), false),
                                                                    
)

# Helper to check interaction
is_interacting(model::AbstractString) = MODEL_REGISTRY[lowercase(model)][5]


"""
    build_hamiltonian(model::AbstractString, L::Integer; mparams="", mparam_dict=Dict()) -> MPO

Build a Hamiltonian MPO by *model name*, validating required model-specific parameters.

Arguments
---------
- `model`      : one of keys in MODEL_REGISTRY (e.g. "aah", "ssh", "uniform")
- `L`          : log of number of sites (Qubits), i.e. `N=2^L`
- `mparams`    : (optional) string `"k=v,k2=v2"` for model-specific params
- `mparam_dict`: (optional) Dict{Symbol,Any} with model params (merged after `mparams`)

Returns
-------
- `H::MPO`

Notes
-----
- Required params are enforced. Any extra keys are passed as keywords.
- Positional arg order comes from the registry entry.
"""
# Common helper (don’t export)
function _build_hamiltonian_impl(model::AbstractString; mparams::AbstractString="", mparam_dict=Dict{Symbol,Any}())
    haskey(MODEL_REGISTRY, lowercase(model)) || error("Unknown model '$model'. Known: $(collect(keys(MODEL_REGISTRY)))")
    return MODEL_REGISTRY[lowercase(model)]
end

# 1D signature: pass L positionally
function build_hamiltonian(model::AbstractString, L::Integer; mparams::AbstractString="", mparam_dict=Dict{Symbol,Any}())
    fn_sym, dim, required_syms, kw_defaults = _build_hamiltonian_impl(model; mparams, mparam_dict)
    dim == 1 || error("Model '$model' expects 2D sizes (Lx, Ly). Use the 2D method build_hamiltonian(model, Lx, Ly; ...).")
    fn = getfield(@__MODULE__, fn_sym)

    # Merge params: string first, then dict overrides
    p = _parse_param_string(mparams)
    for (k,v) in mparam_dict
        p[k] = v
    end

    # Check required (positional-after-L)
    missing = [k for k in required_syms if !haskey(p, k)]
    !isempty(missing) && error("Missing required params for '$model': $(missing). Provided: $(collect(keys(p))).")

    pos = [p[k] for k in required_syms]  # e.g. [:t, :d] -> [t, d]
    extra = Dict(k=>v for (k,v) in p if !(k in required_syms))
    kw_final = (; kw_defaults..., extra...)

    return fn(L, pos...; kw_final...)
end

# 2D signature: pass (Lx, Ly) positionally
function build_hamiltonian(model::AbstractString, Lx::Integer, Ly::Integer; mparams::AbstractString="", mparam_dict=Dict{Symbol,Any}())
    fn_sym, dim, required_syms, kw_defaults = _build_hamiltonian_impl(model; mparams, mparam_dict)
    dim == 2 || error("Model '$model' expects 1D size L. Use the 1D method build_hamiltonian(model, L; ...).")
    fn = getfield(@__MODULE__, fn_sym)

    # Merge params: string first, then dict overrides
    p = _parse_param_string(mparams)
    for (k,v) in mparam_dict
        p[k] = v
    end

    # Check required (positional-after-Lx,Ly)
    missing = [k for k in required_syms if !haskey(p, k)]
    !isempty(missing) && error("Missing required params for '$model': $(missing). Provided: $(collect(keys(p))).")

    pos = [p[k] for k in required_syms]  # e.g. [:t] -> [t]
    extra = Dict(k=>v for (k,v) in p if !(k in required_syms))
    kw_final = (; kw_defaults..., extra...)

    return fn(Lx, Ly, pos...; kw_final...)
end


export build_hamiltonian


function qtt_mpo(L, xvals, sites, func; tol_quantics=1e-8, maxbonddim_quantics=50) 
    qtt  = QTCI.quanticscrossinterpolate(ComplexF64, func, xvals; tolerance=tol_quantics, maxbonddim=maxbonddim_quantics)[1]
    tt   = TCI.tensortrain(qtt.tci)
    mps  = MPS(tt; sites)
    mpo  = outer(mps', mps)
    for s in 1:L
        mpo.data[s] = Quantics._asdiagonal(mps.data[s], sites[s])
    end
    return mpo
end

# -------------------------- Model Hamiltonians -------------------------

# -------------------------- 1D Models -------------------------



"""
    HUniform(L, t; v=0.0, tol_quantics=1e-8, maxbonddim_quantics=10, nn=1, negate=true) -> MPO

Uniform-hopping tight-binding Hamiltonian on `L` Qubit sites with an optional
uniform onsite potential.

- Hopping field is constant (via Quantics TT), then mapped to a kinetic MPO with reach `nn`.
- Onsite term adds `v * ∑_x |x⟩⟨x|` (diagonal MPO).

Required:
- `L::Int` : number of sites
- `t::Real`: hopping amplitude (if `negate=true`, the hopping field uses `-t` to match common TB sign)

Keywords:
- `v::Real = 0.0`                    # uniform onsite potential
- `tol_quantics::Real = 1e-8`         # Quantics cross-interp tolerance
- `maxbonddim_quantics::Int = 10`     # Quantics TT bond cap
- `nn::Int = 1`                       # neighbor reach for kinetic (1 = NN)

Returns:
- `H::MPO`
"""
function HUniform(L::Integer, t; v::Real=1e-6, tol_quantics::Real=1e-8,
                  maxbonddim_quantics::Integer=10, nn::Integer=1)

    if v==0.0
        @warn "onsite potential v needs to be non-zero to avoid failure of TCI"
    end
    
    N = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    # --- Uniform hopping field (constant over sites)
    const_hop(x; t0=t) = t0
    hops_MPO = qtt_mpo(L, xvals, sites, const_hop; tol_quantics=tol_quantics)

    # --- Uniform onsite potential v
    const_onsite(x; v0=v) = v0
    on_MPO = qtt_mpo(L, xvals, sites, const_onsite; tol_quantics=tol_quantics)

    # --- Kinetic + Onsite
    Hkin = kin_builders.kineticNNN(L, sites, hops_MPO, nn)
    H    = +(Hkin, on_MPO; cutoff=1e-8)

    return H
end


"""
    HSSH(L, t, d; tol_quantics=1e-8, maxbonddim_quantics=10, nn=1) -> MPO

Construct the SSH Hamiltonian as an MPO on `L` Qubit sites using a dimerized
hopping pattern:
    SSH_hop(x; t0=t, t1=d) = (x even ? t0+t1 : t0-t1)

The spatial hopping field is encoded as a diagonal MPO (via Quantics TT),
then converted to a kinetic operator with nearest-neighbor reach by default
(using `kin_builders.kineticNNN(..., nn)`).

Required (positional) arguments:
- `L::Int` : number of sites (Qubits)
- `t::Real`: base hopping amplitude
- `d::Real`: dimerization amplitude

Keyword arguments:
- `tol_quantics::Real = 1e-8`          – Quantics cross interpolation tolerance
- `maxbonddim_quantics::Int = 10`      – Quantics TT bond cap (SSH is simple)
- `nn::Int = 1`                        – neighbor reach for `kineticNNN` (1 = NN)

Returns:
- `H::MPO` – SSH Hamiltonian MPO
"""
function HSSH(L::Integer, t, d; tol_quantics::Real=1e-8, maxbonddim_quantics::Integer=10, nn::Integer=1)
    N = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    # SSH hopping: alternating t±d on even/odd sites
    SSH_hop(x; t0=t, t1=d) = (x % 2 == 0) ? (t0 + t1) : (t0 - t1)
    hops_MPO = qtt_mpo(L, xvals, sites, SSH_hop; tol_quantics=tol_quantics)

    # Build kinetic operator with reach `nn` (nn=1 => nearest-neighbor SSH)
    H = kin_builders.kineticNNN(L, sites, hops_MPO, nn)

    return H
end



"""
    HAAH(L, d, DD, tconst; tol_quantics=1e-8, maxbonddim_quantics=50) -> MPO

Construct the Aubry–André–Harper (AAH) Hamiltonian as an `MPO` on `L` Qubit sites,
using Quantics-based diagonal fields and the `kin_builders.kineticNNN` builder
(with `nn=1` for nearest-neighbor hopping).

Required (positional) arguments:
- `L::Int`       : number of sites (Qubits)
- `d::Real`      : AAH onsite amplitude (V)
- `DD::Real`     : AAH phase offset (δ)
- `tconst::Real` : (negative) hopping amplitude t, used via `const_hop(x) = -tconst`

Keyword arguments (defaults match your snippet):
- `tol_quantics=1e-8`          : tolerance for Quantics cross interpolation
- `maxbonddim_quantics=50`     : TT bond cap for Quantics cross interpolation

Returns:
- `H::MPO` the full AAH Hamiltonian MPO
"""

function HAAH(L::Integer, V, phi, t; b=((1 + sqrt(5))/2), tol_quantics::Real=1e-8, maxbonddim_quantics::Integer=50)
    N = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    # ------------------------- potentials (Quantics) -------------------------
    faah(x; V=V, del=phi) = V * cos(2π * b * (x)  + del)
    const_hop(x; t=t) = t
    onsite_MPO = qtt_mpo(L, xvals, sites, faah; tol_quantics=tol_quantics)
    hops_MPO = qtt_mpo(L, xvals, sites, const_hop; tol_quantics=tol_quantics)

    # ------------------------- Hamiltonian -------------------------
    Hhop = kin_builders.kineticNNN(L, sites, hops_MPO, 1)
    Hons = onsite_MPO
    H    = +(Hhop, Hons; cutoff=1e-8)

    return H
end

# -------------------------- 2D Models -------------------------  

"""
    HUniform2Dsquare(Lx::Integer, Ly::Integer, t;
                     tol_quantics::Real = 1e-8,
                     maxbonddim_quantics::Integer = 10,
                     cutoff::Real = 1e-10) -> MPO

Build the **uniform tight-binding Hamiltonian** on a 2D **square lattice** 
represented in a row-major binary basis (Qubit sites, `L = Lx + Ly`).

# Arguments
- `Lx, Ly` : binary exponents defining system size 
             (`Nx = 2^Lx`, `Ny = 2^Ly` ⇒ total sites `N = Nx*Ny`).
- `t`      : uniform hopping amplitude.
- `tol_quantics` : tolerance for Quantics cross interpolation.
- `maxbonddim_quantics` : bond dimension cutoff in Quantics TT compression.
- `cutoff` : MPO summation cutoff (for ITensors `+`).

# Construction
1. Defines a constant hopping field `t` on the 2D grid.
2. Wraps `(x,y)` → `i` for Quantics cross interpolation over the 2D lattice.
3. Builds a **diagonal MPO** for the hopping field from TT → MPS → MPO.
4. Constructs:
   - `Hintra`: intra-row hoppings with row breaks 
               (`kineticintra2DNNN`).
   - `Hinter`: inter-row hoppings coupling neighboring rows 
               (`kineticNNN` with reach `Nx`).
5. Returns total Hamiltonian `Htot = Hintra + Hinter`.

# Returns
- `MPO` representing the uniform 2D square lattice Hamiltonian.
"""
function HUniform2Dsquare(Lx::Integer, Ly::Integer, t;
                    tol_quantics::Real = 1e-8,
                    maxbonddim_quantics::Integer = 10,
                    cutoff::Real = 1e-10)

    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly
    N      = Nx * Ny

    sites = siteinds("Qubit", L)
    xvals = 0:N-1  # linearized 2D grid: i = x + y*Nx

    # Constant 2D hopping field
    const_hop2D(x, y; t0 = t) = t0

    # Wrap (x,y) → linear index i
    wrap2D0(f, Nx) = i -> f((i % Nx), (i ÷ Nx))
    w = wrap2D0((x, y) -> const_hop2D(x, y), Nx)
    hops_MPO = qtt_mpo(L, xvals, sites, w; tol_quantics=tol_quantics)

    # Intra-row and inter-row hoppings
    Hintra = kin_builders.kineticintra2DNNN(Lx, Ly, sites, hops_MPO, 1)
    Hinter = kin_builders.kineticNNN(L,    sites, hops_MPO, Nx)

    # Total Hamiltonian
    Htot = +(Hintra, Hinter; cutoff = cutoff)
    return Htot
end



"""
    Huniform2Dhex(Lx::Integer, Ly::Integer, t; 
                  tol_quantics::Real = 1e-8,
                  maxbonddim_quantics::Integer = 10,
                  cutoff::Real = 1e-10) -> MPO

Build the **uniform tight-binding Hamiltonian** on a 2D **hexagonal lattice** 
represented in a row-major binary basis (Qubit sites, `L = Lx + Ly`).

# Arguments
- `Lx, Ly` : binary exponents defining system size 
             (`Nx = 2^Lx`, `Ny = 2^Ly` ⇒ total sites `N = Nx*Ny`).
- `t`      : uniform hopping amplitude.
- `tol_quantics` : tolerance for Quantics cross interpolation.
- `maxbonddim_quantics` : bond dimension cutoff in Quantics TT compression.
- `cutoff` : MPO summation cutoff (for ITensors `+`).

# Construction
1. Defines a constant hopping field `t` on the 2D grid.
2. Wraps `(x,y)` → `i` for Quantics cross interpolation over the 2D lattice.
3. Builds a **diagonal MPO** for the hopping field from TT → MPS → MPO.
4. Constructs:
   - `Hintra`: intra-row hoppings with hexagonal row-breaking rules 
               (`kineticintra2DNNhex`).
   - `Hinter`: inter-row hoppings coupling neighboring rows (`kineticNNN`).
5. Returns total Hamiltonian `Htot = Hintra + Hinter`.

# Returns
- `MPO` representing the uniform 2D hexagonal lattice Hamiltonian.
"""
function HUniform2Dhex(Lx::Integer, Ly::Integer, t;
                    tol_quantics::Real = 1e-8,
                    maxbonddim_quantics::Integer = 10,
                    cutoff::Real = 1e-10)

    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly
    N      = Nx * Ny

    sites = siteinds("Qubit", L)
    xvals = 0:N-1  # linearized 2D grid: i = x + y*Nx

    # Constant 2D hopping field
    const_hop2D(x, y; t0 = t) = t0

    # Wrap (x,y)->... to a 1D function of linear index i
    wrap2D0(f, Nx) = i -> f((i % Nx), (i ÷ Nx))
    w = wrap2D0((x, y) -> const_hop2D(x, y), Nx)
    hops_MPO = qtt_mpo(L, xvals, sites, w; tol_quantics=tol_quantics)

    # Intra-row hopping with row breaks; inter-row hopping couples across rows
    Hintra = kin_builders.kineticintra2DNNhex(Lx, Ly, sites, hops_MPO, 1)
    Hinter = kin_builders.kineticNNN(L,    sites, hops_MPO, Nx)

    # Total Hamiltonian
    Htot = +(Hintra, Hinter; cutoff = cutoff)
    return Htot
end

## Triangular is not really working yet so I have to modify it 
"""
    Huniform2Dtri(Lx::Integer, Ly::Integer, t; 
                  tol_quantics::Real = 1e-8,
                  maxbonddim_quantics::Integer = 10,
                  cutoff::Real = 1e-10) -> MPO

Build the **uniform tight-binding Hamiltonian** on a 2D **triangular lattice** 
represented in a row-major binary basis (Qubit sites, `L = Lx + Ly`).

# Arguments
- `Lx, Ly` : binary exponents defining system size 
             (`Nx = 2^Lx`, `Ny = 2^Ly` ⇒ total sites `N = Nx*Ny`).
- `t`      : uniform hopping amplitude.
- `tol_quantics` : tolerance for Quantics cross interpolation.
- `maxbonddim_quantics` : bond dimension cutoff in Quantics TT compression.
- `cutoff` : MPO summation cutoff (for ITensors `+`).

# Construction
1. Defines a constant hopping field `t` on the 2D grid.
2. Wraps `(x,y)` → `i` for Quantics cross interpolation.
3. Builds a **diagonal MPO** for the hopping field from TT → MPS → MPO.
4. Constructs:
   - `HintraNN`  : nearest-neighbor intra-row hoppings (`kineticintra2DNNN`).
   - `HinterNN1` : inter-row hoppings along SW↗NE diagonals 
                   (`kineticinterNNNtriSWNE`).
   - `HinterNN2` : inter-row hoppings along SE↖NW diagonals 
                   (`kineticinterNNNtriSENW`).
5. Returns total Hamiltonian `Htot = HintraNN + HinterNN1 + HinterNN2`.

# Returns
- `MPO` representing the uniform 2D triangular lattice Hamiltonian.
"""
function HUniform2Dtri(Lx::Integer, Ly::Integer, t;
                 tol_quantics::Real = 1e-8,
                 maxbonddim_quantics::Integer = 10,
                 cutoff::Real = 1e-10)

    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly
    N      = Nx * Ny

    sites = siteinds("Qubit", L)
    xvals = 0:N-1  # linearized index i = x + y*Nx

    # Helper: wrap linear i to (x,y) for Quantics cross-interpolation
    wrap2D(f) = i -> begin
        x = Int(mod(i, Nx))
        y = Int(fld(i, Nx))
        f(x, y)
    end

    # Constant hopping
    const_hop2D(x, y) = t

    # Intra-row field
    w1 = wrap2D(const_hop2D)
    hops_MPO = qtt_mpo(L, xvals, sites, w1; tol_quantics=tol_quantics)

    # Kinetic pieces
    HintraNN   = kin_builders.kineticintra2DNNN(Lx, Ly, sites, hops_MPO,  1)
    HinterNN1  = kin_builders.kineticinterNNNtriSWNE(Lx, Ly, sites, hops_MPO, Nx + 1)
    HinterNN2  = kin_builders.kineticinterNNNtriSENW(Lx, Ly, sites, hops_MPO, Nx - 1)

    # Total Hamiltonian
    Htot = +(HintraNN, HinterNN1; cutoff = cutoff) 
    Htot = +(Htot,     HinterNN2; cutoff = cutoff)

    return Htot
end



"""
    HChern8(Lx, Ly, V, t;
            a = 5/64 * 2^Lx,
            t2 = 0.2*t,
            tol_quantics::Real = 1e-8,
            maxbonddim_quantics::Integer = 10,
            cutoff::Real = 1e-10) -> MPO

Build the 8-fold “Chern mosaic” Hamiltonian on an `Lx × Ly` grid (flattened row-major).
Let `Nx = 2^Lx`, `Ny = 2^Ly`, `L = Lx + Ly`, `N = Nx*Ny`, and `sites = siteinds("Qubit", L)`.

Fields (all diagonal as MPOs via Quantics/TCI):
- `alt_hop_x(x) = (-1)^( (x+1) mod Nx ) * t`           — alternating along x only
- `const_hop2D(x,y) = t`                                — constant intra-row hopping
- `func8fold(x,y)`                                      — 8-fold modulation from 4 rotated k-vectors
- Two modulated alternations for the diagonal inter-row terms:
    * `w2(x,y) = alt_hop_x(x-1) * func8fold(x,y)`
    * `w3(x,y) = alt_hop_x(x)   * func8fold(x,y)`

Kinetic composition (using kin_builders):
- `HinterNN   = kin_builders.kineticNN(L, sites, hops_MPO,   Nx)`          # inter-row NN
- `HintraNN   = kin_builders.kineticintra2DNNN(Lx, Ly, sites, hops_MPO1, 1)`# intra-row NN
- `HinterNNN  = kin_builders.kineticinterNNNSWNE(Lx, Ly, sites, hops_MPO2, Nx+1)`  # SW↗NE diag
- `HinterNNN2 = kin_builders.kineticinterNNNSENW(Lx, Ly, sites, hops_MPO3, Nx-1)`  # SE↖NW diag

Returns `Htot = HinterNN + HinterNNN + HinterNNN2 + HintraNN` with `cutoff` applied to sums.

Notes
-----
- Fixed: earlier calls `func8fold(x,y,N,V)` were missing `a,t2`; now `func8fold(x,y,a,V,t2)`.
- Unified wrapping: `wrap2D` maps linear index `i ∈ 0:N-1` → `(x=i % Nx, y=i ÷ Nx)`.
- `alt_hop_x` is defined on the x-coordinate (0..Nx-1), not the linear index.
"""
function HChern8(Lx::Integer, Ly::Integer, V, t;
                 a = 5/64 * 2^Lx,
                 t2 = 0.2*t,
                 tol_quantics::Real = 1e-8,
                 maxbonddim_quantics::Integer = 10,
                 cutoff::Real = 1e-10)

    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly
    N      = Nx * Ny

    sites = siteinds("Qubit", L)
    xvals = 0:N-1  # linearized index i = x + y*Nx

    # --- Alternating hopping along x (x is 0..Nx-1)
    alt_hop_x(x) = (-1)^mod(x + 1, Nx) * t

    # --- 8-fold modulation term on the 2D coordinates
    function func8fold(x, y, a, V, t2)
        # 4 vectors: (2π/a) * e_x, e_y, and their 45° rotation
        Ka1 = (2π / a) .* ( [1.0, 0.0] )
        Kb1 = (2π / a) .* ( [0.0, 1.0] )
        θ   = deg2rad(45.0)
        Rt  = [cos(θ)  sin(θ);
               -sin(θ) cos(θ)]
        Ka2 = Rt * Ka1
        Kb2 = Rt * Kb1
        K   = (Ka1, Kb1, Ka2, Kb2)
        xy  = [x, y]
        F   = 0.0 + 0.0im
        @inbounds for k in K
            F += 1im * V * t2 * (cos(dot(k, xy))^2)
        end
        return F
    end

    # --- Helper: wrap linear i to (x,y) for Quantics cross-interpolation
    wrap2D(f) = i -> begin
        x = Int(mod(i, Nx))
        y = Int(fld(i, Nx))
        f(x, y)
    end

    # diagonal bond midpoints
    mid_SWNE(x, y) = (x + 0.5, y + 0.5)  # (x,y) ↔ (x+1,y+1)
    mid_SENW(x, y) = (x - 0.5, y + 0.5)  # (x,y) ↔ (x-1,y+1)


    # Constant intra-row hopping
    const_hop2D(x, y) = t

    # The two diagonally modulated alternations (staggered by x → x-1)
    w1 = wrap2D(const_hop2D)                                   # intra-row

    # Staggering along x (keep your choices for the alternating factor)
    w2 = wrap2D((x,y) -> begin
        xm, ym = mid_SWNE(x, y)
        alt_hop_x(mod(x - 1, Nx)) * func8fold(xm, ym, a, V, t2)   # SW↗NE branch
    end)
    w3 = wrap2D((x,y) -> begin
        xm, ym = mid_SENW(x, y)
        alt_hop_x(x) * func8fold(xm, ym, a, V, t2)                # SE↖NW branch
    end)


    # Also need a pure x-alternating field extended over the 2D grid
    w_alt = wrap2D((x,y) -> alt_hop_x(x))

    # --- Quantics cross interpolation (use provided tolerance/bond dims)
    hops_MPO  = qtt_mpo(L, xvals, sites, w_alt; tol_quantics=tol_quantics,   maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO1 = qtt_mpo(L, xvals, sites, w1;    tol_quantics=tol_quantics,   maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO2 = qtt_mpo(L, xvals, sites, w2;    tol_quantics=tol_quantics,   maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO3 = qtt_mpo(L, xvals, sites, w3;    tol_quantics=tol_quantics,   maxbonddim_quantics=maxbonddim_quantics)


    # --- Kinetic pieces 
    HinterNN   = kin_builders.kineticNNN(            L,    sites, hops_MPO,   Nx)
    HintraNN   = kin_builders.kineticintra2DNNN(   Lx, Ly, sites, hops_MPO1,  1)
    HinterNNN  = kin_builders.kineticinterNNNSWNE( Lx, Ly, sites, hops_MPO2, Nx + 1)
    HinterNNN2 = kin_builders.kineticinterNNNSENW( Lx, Ly, sites, hops_MPO3, Nx - 1)

    # --- Total Hamiltonian
    Htot = +(HinterNN,  HinterNNN;  cutoff = cutoff)
    Htot = +(Htot,      HinterNNN2; cutoff = cutoff)
    Htot = +(Htot,      HintraNN;   cutoff = cutoff)

    return Htot
end


"""
    H2DChernhex(Lx::Integer, Ly::Integer, t;
                t2 = 0.2*t,
                tol_quantics::Real = 1e-8,
                maxbonddim_quantics::Integer = 10,
                cutoff::Real = 1e-10) -> MPO

Build the **Chern insulator Hamiltonian** on a 2D **hexagonal lattice** 
encoded in a row-major binary basis (`L = Lx + Ly` qubits, 
`Nx = 2^Lx`, `Ny = 2^Ly`, total sites `N = Nx*Ny`).

# Arguments
- `Lx, Ly` : binary exponents for system dimensions 
             (`Nx = 2^Lx`, `Ny = 2^Ly`).
- `t`      : uniform nearest-neighbor hopping amplitude.
- `t2`     : complex next-nearest neighbor hopping amplitude 
             (defaults to `0.2*t`). Implemented as a checkerboard 
             pattern ±i·t2 depending on the parity of (x+y).
- `tol_quantics` : tolerance for Quantics cross interpolation.
- `maxbonddim_quantics` : maximum bond dimension for TT compression.
- `cutoff` : cutoff tolerance for MPO summations (`+`).

# Construction
1. Define two diagonal MPO fields via Quantics cross interpolation:
   - `hops_MPO`      : constant hopping `t`.
   - `hops_MPOalter` : staggered ±i·t2 “checkerboard” alternation.
2. Diagonalize both fields (TT → MPS → diagonal MPO).
3. Build kinetic terms:
   - `Hintra`  : intra-row hoppings with row breaks 
                 (`kineticintra2DNNhex`).
   - `Hinter`  : vertical inter-row hoppings (`kineticNNN` with reach Nx).
   - `HNNinter1`: long-range vertical NNN hopping (2Nx).
   - `HNNinter2`: diagonal SW↗NE inter-row hoppings (`kineticinterNNNSWNE`).
   - `HNNinter3`: diagonal SE↖NW inter-row hoppings (`kineticinterNNNSENW`).
4. Sum all pieces with the given cutoff.

# Returns
- `MPO` representing the hexagonal Chern Hamiltonian with uniform `t`
  hoppings and complex next-nearest neighbor hoppings `±i·t2`.
"""
function H2DChernhex(Lx::Integer, Ly::Integer, t; 
                     t2 = 0.2*t,
                     tol_quantics::Real = 1e-8,
                     maxbonddim_quantics::Integer = 10,
                     cutoff::Real = 1e-10)

    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly
    N      = Nx * Ny

    sites = siteinds("Qubit", L)
    xvals = 0:N-1  # linearized 2D index: i = x + y*Nx

    # --- Hopping fields -----------------------------------------------------

    # Constant hopping t
    const_hop2D(x, y; t0 = t) = t0

    # Checkerboard alternation: ±i·t2 depending on parity of x+y
    alt_hop_xy(x, y) = isodd(x + y) ? -1im*t2 : 1im*t2

    # Wrap (x,y) → linear index i for Quantics interpolation
    wrap2D(f) = i -> begin
        x = Int(mod(i, Nx))
        y = Int(fld(i, Nx))
        f(x, y)
    end
    w  = wrap2D((x, y) -> const_hop2D(x, y))
    w1 = wrap2D((x, y) -> alt_hop_xy(x, y))


    # --- Quantics MPOs ------------------

    hops_MPOalter = qtt_mpo(L, xvals, sites, w1; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO      = qtt_mpo(L, xvals, sites, w;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    # --- Kinetic terms -----------------------------------------------------
    Hintra   = kin_builders.kineticintra2DNNhex(Lx, Ly, sites, hops_MPO, 1)
    Hinter   = kin_builders.kineticNNN(L,       sites, hops_MPO, Nx)

    HNNinter1 = kin_builders.kineticNNN(L,       sites, hops_MPOalter, 2*Nx) 
    HNNinter2 = kin_builders.kineticinterNNNSWNE(Lx, Ly, sites, hops_MPOalter, Nx + 1)
    HNNinter3 = kin_builders.kineticinterNNNSENW(Lx, Ly, sites, hops_MPOalter, Nx - 1)

    # --- Total Hamiltonian -------------------------------------------------
    Htot = +(Hintra, Hinter;   cutoff = cutoff)
    Htot = +(Htot,   HNNinter1; cutoff = cutoff)
    Htot = +(Htot,   HNNinter2; cutoff = cutoff)
    Htot = +(Htot,   HNNinter3; cutoff = cutoff)

    return Htot
end


function HQC2Dsquare(Lx::Integer, Ly::Integer, t::Real = 1.0;
                  tol_quantics::Real = 1e-8,
                  maxbonddim_quantics::Integer = 100, 
                  cutoff::Real = 1e-10) 

    # System sizes: Nx, Ny sites in each direction (powers of 2)
    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly           # number of qubits in TT representation
    N      = Nx * Ny           # total number of physical sites

    sites = siteinds("Qubit", L)
    xvals = 0:N-1               # linearized 2D grid indices

        # --- 8-fold modulation term on the 2D coordinates
    function func8fold(x, y, V;  Nx=Nx)
        # 4 vectors: (2π/a) * e_x, e_y, and their 45° rotation
        a = 1          # lattice constant for 8-fold modulation

        b1 = (5*sqrt(5)*a/2) # atomic scale wavevector
        b2 = (sqrt(3)*(Nx*a/16))  # superlattice scale wavevector

        Ka1 = 2π .* ( [1.0, 0.0] )
        Kb1 = 2π .* ( [0.0, 1.0] )
        tht   = deg2rad(45.0)
        Rt  = [cos(tht)  sin(tht);
                -sin(tht) cos(tht)]
        Ka2 = Rt * Ka1
        Kb2 = Rt * Kb1
        K   = (Ka1, Kb1, Ka2, Kb2)
        xy  = [x - Nx/2, y - Nx/2] # the offset is to obtain a nice symmetric pattern (assumes Nx=Ny)
        
        cosines = 0.0
        for k in K
            cosines += (2.5*cos(dot(k, xy)/b1) + cos(dot(k, xy)/b2)) #same as yitao
        end
        return V * (1 + 0.1 * cosines)

    end

    # Wrappers that evaluate f at the bond midpoints on square lattice (for hoppings)
    wrap2D_mid_x(f, Nx) = i -> begin
        x = i % Nx
        y = i ÷ Nx
        f(x + 0.5, y)          # midpoint between (x,y) and (x+1,y)
    end

    wrap2D_mid_y(f, Nx) = i -> begin
        x = i % Nx
        y = i ÷ Nx
        f(x, y + 0.5)          # midpoint between (x,y) and (x,y+1)
    end

    # Intra-row and inter-row nearest-neighbor hoppings at midpoints
    intra_row_hop = wrap2D_mid_x((x, y) -> func8fold(x, y, t; Nx=Nx), Nx)
    inter_row_hop = wrap2D_mid_y((x, y) -> func8fold(x, y, t; Nx=Nx), Nx)


    # Quantics cross interpolation of the hopping fields
    hops_MPOintra = qtt_mpo(L, xvals, sites, intra_row_hop; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPOinter = qtt_mpo(L, xvals, sites, inter_row_hop; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    # Build kinetic Hamiltonians
    Hintra = kin_builders.kineticintra2DNNN(Lx, Ly, sites, hops_MPOintra, 1)
    Hinter = kin_builders.kineticNNN(L,    sites, hops_MPOinter, Nx)

    # Total non-interacting Hamiltonian
    H0 = +(Hinter,  Hintra;  cutoff = cutoff)

    return H0
end



end # module Hamiltonians