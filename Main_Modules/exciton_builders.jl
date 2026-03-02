module exciton_builders


# ---------------------------------------------------------------------
# Dependencies 
# ---------------------------------------------------------------------
import TensorCrossInterpolation as TCI
import QuanticsTCI as QTCI
using ITensors
using ITensorMPS
using ITensorMPS: MPO, MPS, OpSum, expect, inner, siteinds
using LinearAlgebra
using QuanticsTCI
using Quantics


include("kin_builders.jl")
using .kin_builders
include("extra_util.jl")
using .extra_util
include("Hamiltonians.jl")
using .Hamiltonians

# ---------------------------------------------------------------------
function Exciton_2D(Lx, Ly, Ufunc, hop, on_site; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-8, maxbonddim_gen=200)
    L = Lx + Ly
    Nphys  = 2^L
    xvals  = 0:(Nphys - 1)

    kinetic_mpo = hopping_mpo_exciton_2D(Lx, Ly, hop, on_site; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    sites_eh = getindex.(siteinds(kinetic_mpo),2)

    interaction_operator = build_interaction_op_exciton(L, sites_eh, Ufunc)

    H_exciton = kinetic_mpo + interaction_operator

    return H_exciton
end


function hopping_mpo_exciton_2D(Lx, Ly, hop, on_site; tol_quantics=1e-8, maxbonddim_quantics=100) 
    L = Lx + Ly
    Nphys  = 2^L
    xvals  = 0:(Nphys - 1)

    # physical chain sites and kinetic MPO on that chain

    Ham_c  = Hamiltonians.build_hamiltonian("uniform2dsquare", Lx, Ly; mparam_dict=Dict(:t=>hop)) #uniform hopping Hamiltonian on 2D square lattice (electron)

    sites = getindex.(siteinds(Ham_c), 2)

    on_site_MPO_c  = qtt_mpo(L, xvals, sites, on_site;
                                         tol_quantics=tol_quantics,
                                         maxbonddim_quantics=maxbonddim_quantics)
                               

    Ham_v  = Hamiltonians.build_hamiltonian("uniform2dsquare", Lx, Ly; mparam_dict=Dict(:t=>hop)) #uniform hopping Hamiltonian on 2D square lattice (hole)
    sites = getindex.(siteinds(Ham_v), 2)

    on_site_MPO_v  = qtt_mpo(L, xvals, sites, on_site;
                                         tol_quantics=tol_quantics,
                                         maxbonddim_quantics=maxbonddim_quantics)

    H_free_c = Ham_c + on_site_MPO_c      
    H_free_v = Ham_v - on_site_MPO_v  
    
    v = zip(siteinds("Qubit", L), siteinds("Qubit", L)) |> collect
    sites_eh = collect(Iterators.flatten(v))
                                        
    KIN_TOT = extra_util.interleave_mpo(H_free_c, sites_eh, 0) - extra_util.interleave_mpo(H_free_v, sites_eh, 1)

    return KIN_TOT
end



function Exciton_1D(L, sites, Ufunc, hoppingfunc, on_site; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-8, maxbonddim_gen=200)
    Nphys  = 2^L
    xvals  = 0:(Nphys - 1)

    kinetic_mpo = hopping_mpo_exciton(L, sites, hoppingfunc, on_site; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    sites_eh = getindex.(siteinds(kinetic_mpo),2)

    interaction_operator = build_interaction_op_exciton(L, sites_eh, Ufunc)

    H_exciton = kinetic_mpo + interaction_operator

    return H_exciton
end


function qtt_mpo(L, xvals, sites, func; tol_quantics=1e-8, maxbonddim_quantics=100) 
    qtt  = QTCI.quanticscrossinterpolate(ComplexF64, func, xvals; tolerance=tol_quantics, maxbonddim=maxbonddim_quantics)[1]
    tt   = TCI.tensortrain(qtt.tci)
    mps  = MPS(tt; sites)
    mpo  = outer(mps', mps)
    for s in 1:L
        mpo.data[s] = Quantics._asdiagonal(mps.data[s], sites[s])
    end
    return mpo
end



function hopping_mpo_exciton(L, sites, hoppingfunc, on_site; tol_quantics=1e-8, maxbonddim_quantics=100)
    Lphys  = L 
    Nphys  = 2^Lphys
    xvals  = 0:(Nphys - 1)

    # physical chain sites and kinetic MPO on that chain

    hops_MPO_c  = qtt_mpo(Lphys, xvals, sites, hoppingfunc;
                                         tol_quantics=tol_quantics,
                                         maxbonddim_quantics=maxbonddim_quantics)
    k_mpo_c     = kin_builders.kineticNNN(Lphys, sites, hops_MPO_c, 1) # MPO on first block of sites

    on_site_MPO_c  = qtt_mpo(Lphys, xvals, sites, on_site;
                                         tol_quantics=tol_quantics,
                                         maxbonddim_quantics=maxbonddim_quantics)
                               

    hops_MPO_v  = qtt_mpo(Lphys, xvals, sites, hoppingfunc;
                                         tol_quantics=tol_quantics,
                                         maxbonddim_quantics=maxbonddim_quantics)
    k_mpo_v     = kin_builders.kineticNNN(Lphys, sites, hops_MPO_v, 1) # MPO on second block of sites


    on_site_MPO_v  = qtt_mpo(Lphys, xvals, sites, on_site;
                                         tol_quantics=tol_quantics,
                                         maxbonddim_quantics=maxbonddim_quantics)

    H_free_c = k_mpo_c + on_site_MPO_c      
    H_free_v = k_mpo_v - on_site_MPO_v  
    
    v = zip(siteinds("Qubit", L), siteinds("Qubit", L)) |> collect
    sites_eh = flat = collect(Iterators.flatten(v))
                                        
    KIN_TOT = extra_util.interleave_mpo(H_free_c, sites_eh, 0) - extra_util.interleave_mpo(H_free_v, sites_eh, 1)

    return KIN_TOT
end


function build_interaction_op_exciton(L, sites, Ufunc)

    evals = range(1,  (2^(L ) ), length=2^(L ))
    hvals = range(1,  (2^(L ) ), length=2^(L ))

    o(x, y) = x == y ? Ufunc(x) : 0

    qtt = quanticscrossinterpolate(Float64, o,  [evals, hvals]; tolerance=1e-8)[1]
    tt = TCI.tensortrain(qtt.tci)
    int_mps = MPS(tt)

    int_mpo = - extra_util.mps_to_diagonal_mpo(int_mps, sites)  #This one is without the one body contribution

    return int_mpo
end




# --------------------------- build the full exciton Hamiltonian as a dense matrix for cross reference -------------------------------
function HExcitonDense_build(N, Ufunc, hopssh; on_site=0.5) 

    HU = zeros(Float64, N^2, N^2)
    HT = zeros(Float64, N^2, N^2)

    # Kronecker delta
    delta(i, j) = i == j ? 1.0 : 0.0

    # interaction part
    U  = [Ufunc(i) for i in 0:N-1]
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        r = (i - 1) * N + j
        c = (k - 1) * N + l
        # HU[r, c] = U[i] * (delta(i, k) * delta(j, l) -
        #                 delta(i, l) * delta(j, k) * delta(j, l)) 
        HU[r, c] = -U[i] * (delta(i, l) * delta(j, k) * delta(j, l)) # without the one body term
                        
    end

    # hopping amplitudes 
    hops = [hopssh(i) for i in 0:N-2]


    hoppingc = (diagm(1 => hops) + diagm(-1 => hops)) .+ on_site*Matrix{Float64}(I, N, N)
    hoppingv = (diagm(1 => hops) + diagm(-1 => hops)) .- on_site*Matrix{Float64}(I, N, N)

    # kinetic part
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        r = (i - 1) * N + j
        c = (k - 1) * N + l
        HT[r, c] += hoppingc[i, k] * delta(j, l) -
                    hoppingv[j, l] * delta(i, k) 
    end

    return HT + HU
end



end # module