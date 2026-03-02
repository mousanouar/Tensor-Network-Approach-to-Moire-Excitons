module ExcitonKPM

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
using Statistics
using CUDA 
using FFTW


# Load your module (must be in the same directory)
include("kin_builders.jl")
using .kin_builders


# ---------------------------------------------------------------------
# Operator definitions 
# ---------------------------------------------------------------------

"""
ITensors operator for σ⁺ on a Qubit site.
"""
ITensors.op(::OpName"sigma_plus",::SiteType"Qubit") =
 [0 1
  0 0]

"""
ITensors operator for σ⁻ on a Qubit site.
"""
ITensors.op(::OpName"sigma_minus",::SiteType"Qubit") =
 [0 0
  1 0]

# ---------------------------------------------------------------------
# Utilities 
# ---------------------------------------------------------------------


function to_binary_vector(n, size)
    # Convert the integer n to a binary string (without leading zeros)
    binary_str = string(n, base=2)
    
    # Pad the binary string with leading zeros on the left
    # so that its length matches the desired size
    padded_binary_str = lpad(binary_str, size, '0')
    
    # Convert the padded string into a vector of characters,
    # then map each character to a proper String ("0" or "1")
    return collect(padded_binary_str) |> x -> map(s -> string(s), x)
end

# exciton basis |x, x>

function mpsexciton(x,sites)
    L = length(sites)
    LPhys = div(L, 2)
    elec = to_binary_vector(Int(x), LPhys)
    hole = to_binary_vector(Int(x), LPhys)

    elechole = Vector{String}(undef, L)
    for i in 1:LPhys
        elechole[2i - 1] = elec[i]
        elechole[2i]     = hole[i]
    end

    excitonMPS = MPS(sites, elechole)

    return excitonMPS
end


function ldos_exc_KPM_Tn(H::MPO, N::Int64, X; cutoff=1e-9, maxdim=200)
     # Kernel Polynomial Method for computing the Tn polynomials from  H at a given X for excitons
     # H is the rescaled Hamiltonian MPO, rescaled, N is the number of Tn polynomials to compute

    apply_kwargs = (cutoff=cutoff, maxdim=maxdim)
    sites = getindex.(siteinds(H), 2)
    L = length(H)  # Number of spins 
    LPhys = div(L, 2) # Number of physical sites (electrons or holes)


    # --- Chebyshev seeds ---
    T_k_minus_2 = mpsexciton(X, sites)  # T₀
    mu_1 = inner(mpsexciton(X, sites)', T_k_minus_2)
    T_k_minus_1 = apply(H, T_k_minus_2; apply_kwargs...)   # T₁
    mu_2 = inner(mpsexciton(X, sites)', T_k_minus_1)
    mun_list = [mu_1, mu_2]

    # --- Recurrence: T_k = 2 Ĥ T_{k-1} - T_{k-2} ---
    for k in 3:N
        # Apply Ĥ to T_{k-1} and form the recurrence with truncation
        T_k = +(2 * apply(H, T_k_minus_1; apply_kwargs...),
                -T_k_minus_2; apply_kwargs...)

        mu_k = inner(mpsexciton(X, sites)', T_k)
        # Shift the window and store mu_k
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k

        push!(mun_list, mu_k)

    end

    return mun_list
end

#Using the HODC Kernel for improved LDOS reconstruction from KPM moments instead of Jackson damping (see paper "High-order orthogonal polynomial kernel for spectral density estimation" by Weisse et al, PRB 2024)
# === Step 1: Compute generic HODC parameters (zl, wl) ===
# These parameters depend only on the order m and are independent of the Hamiltonian
function compute_hodc_params(m=6)
    xl = range(-2.5, 2.5, length=m)
    zl = xl .+ 1im
    # Construct Vandermonde matrix
    A = [z^k for k in 0:m-1, z in zl]
    b = zeros(ComplexF64, m)
    b[1] = 1.0
    wl = A \ b
    return zl, wl
end

# === Step 2: Compute HODC expansion coefficients at a given energy e ===
function get_hodc_weights(e_target, N, eta, zl, wl)
    # Sample the HODC kernel on Chebyshev nodes
    j = 0:N-1
    nodes = cos.(π .* (j .+ 0.5) ./ N)
    
    # Compute K_eta(e_target, x_node) -> paper Eq. 15
    kernel_vals = map(nodes) do x
        term = sum(wl ./ (e_target - x .+ eta .* zl))
        return -1.0/π * imag(term)
    end
    
    # Obtain Chebyshev coefficients nu_k via DCT-II
    # Note: FFTW.REDFT10 corresponds to DCT-II
    nu = FFTW.r2r(kernel_vals, FFTW.REDFT10) ./ N
    nu[1] /= 2.0  # 0th-order term correction
    return nu
end

# === Step 3: Reconstruct get_ldos function at specific normalized energy e in (-0.99, 0.99) and number of moments N===
function get_ldos_hodc(N, mus_raw, eval; eta=0.02, m_order=6) 
    ldos_matrix = 0.0
    
    # Precompute HODC parameters
    zl, wl = compute_hodc_params(m_order)
    
    
    # Compute HODC weight coefficients for the current energy point
    # These nu_k replace the original jackson_kernel[k] * cos(...) / sqrt(...)
    nu_k = get_hodc_weights(eval, N, eta, zl, wl)
    
    # According to paper Eq. 11: LDOS(y) ≈ Σ nu_k * mu_k
    # Considering the standard Chebyshev summation form:
    # nu_0*mu_0 + 2*Σ nu_k*mu_k
    val = mus_raw[1] * nu_k[1]
    for k in 2:N
        val += 2.0 * mus_raw[k] * nu_k[k]
    end
    ldos_matrix = val

    return ldos_matrix
end


function get_ldos_from_mun(mun_list,Nmu,E) #needs normalized energies E # output needs to be divided by the amount of sites N=2^L to get the actual density of states

    # Jackson damping kernel coefficients g_n, n = 0:(Nmu-1)
    jackson_kernel = [(Nmu - n ) * cos(π * n / (Nmu)) + sin(π * n / (Nmu)) / tan(π / (Nmu)) for n in 0:Nmu-1]/Nmu

    # Chebyshev basis factor T_{n-1}(E) = cos((n-1) * arccos(E))
    function G_n(n)
        return cos((n-1)*acos(E))
    end

    # Chebyshev/Jackson expansion of the DOS operator A(E)
    Aq = mun_list[1] * G_n(1) * jackson_kernel[1]      # T₀ term (no factor 2)
    for n in 2:Nmu
        Aq = Aq  +  2 *  mun_list[n] * G_n(n) * jackson_kernel[n] # This is the Chebyshev sum with factor 2 for n≥2
    end

    # Kernel polynomial normalization 
    Aq /= (π * sqrt(1-E^2)) # Normalization of the density MPO

    return  real(Aq)
end


end # module 