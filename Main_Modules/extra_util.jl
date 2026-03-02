module extra_util

using ITensors
using ITensorMPS
using LinearAlgebra
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using Quantics
using LinearAlgebra
using FFTW

################################################################################
#The shift MPO, input as q, output an MPO Q do Qf(x) = f(x+q)
function build_shift_mpo(sites, q)
    N = length(sites)

    # Binary representation of q:
    # site 1 corresponds to the most significant bit (MSB),
    # site N corresponds to the least significant bit (LSB)
    q_bits = [(q >> (N - i)) & 1 for i in 1:N]

    # Link indices with dimension 2:
    # link = 1 -> carry = 1
    # link = 2 -> carry = 0
    links = [Index(2, "Link,l$n") for n in 0:N+1]

    mpo = MPO(sites)

    # Build the MPO from right (LSB) to left (MSB),
    # so that the carry propagates from LSB -> MSB
    for n in N:-1:1 
        s = sites[n]

        # Input carry comes from the right, output carry goes to the left
        l_in  = links[n+1]
        l_out = links[n]

        # Local tensor implementing one-bit binary addition with carry
        T = ITensor(s', s, l_in, l_out)

        # Bit of q associated with site n
        qn = q_bits[n]

        # Enumerate all possible input configurations
        for cin in 0:1        # input carry
            for s_val in 0:1  # local bit value

                # Binary addition rule
                total   = s_val + qn + cin
                res_val = total % 2        # resulting bit
                cout    = total ÷ 2        # output carry

                # Map carry values to link indices
                li = (cin  == 1) ? 1 : 2
                lo = (cout == 1) ? 1 : 2

                # Define the MPO tensor element
                T[s' => (res_val + 1),
                  s  => (s_val + 1),
                  l_in  => li,
                  l_out => lo] = 1.0
            end
        end

        mpo[n] = T
    end

    # Boundary condition at the least significant bit:
    # no incoming carry
    mpo[N] *= onehot(links[N+1] => 2)

    # Boundary condition at the most significant bit:
    # allow both carry = 0 and carry = 1 (cyclic addition modulo 2^N)
    # remove one to make it non-cyclic?
    mpo[1] *= (onehot(links[1] => 1) + onehot(links[1] => 2)) #cyclic addition
    # mpo[1] *= onehot(links[1] => 2) #non-cyclic addition

    return mpo
end
################################################################################


################################################################################
#The HODC kernel to get dos and ldos
# === Step 1: Compute universal HODC parameters (zl, wl) ===
# These parameters depend only on the expansion order m,
# and are independent of the Hamiltonian
function compute_hodc_params(m=6)
    xl = range(-2.5, 2.5, length=m)
    zl = xl .+ 1im
    # Construct the Vandermonde matrix
    A = [z^k for k in 0:m-1, z in zl]
    b = zeros(ComplexF64, m)
    b[1] = 1.0
    wl = A \ b
    return zl, wl
end

# === Step 2: Compute HODC expansion coefficients at a target energy y ===
function get_hodc_weights(y_target, N, eta, zl, wl)
    # Sample the HODC kernel on Chebyshev nodes
    j = 0:N-1
    nodes = cos.(π .* (j .+ 0.5) ./ N)
    
    # Compute K_eta(y_target, x_node) -> see Eq. (15) in the paper
    kernel_vals = map(nodes) do x
        term = sum(wl ./ (y_target - x .+ eta .* zl))
        return -1.0/π * imag(term)
    end
    
    # Obtain Chebyshev coefficients nu_k via DCT-II
    # Check the para under Eq. 17 of the paper for why choosing Chebyshev nodes
    # Note: FFTW.REDFT10 corresponds to DCT-II
    nu = FFTW.r2r(kernel_vals, FFTW.REDFT10) ./ N
    nu[1] /= 2.0  # Correction for the zeroth-order term
    return nu
end

# === Step 3: Modify the moment computation function (remove Jackson kernel) ===
function get_mus_raw(tn_lis)
    # Only keep the raw traces, without multiplying by the Jackson kernel
    return [real(tr(tns)) for tns in tn_lis]
end

# === Step 4: Reconstruct the get_ldos function using HODC ===

"""
    compute_dos_ldos_hodc(
        N,
        tn_lis,
        en_num;
        eta = 0.02,
        m_order = 6
    )

Compute DOS (scalar) and energy-resolved LDOS MPOs using HODC,
based on Chebyshev tensor-network moments.

Inputs:
- N        : Chebyshev expansion order
- tn_lis   : length-N list of MPO/TN objects (Chebyshev moments)
- en_num   : number of energy grid points
- eta      : broadening parameter (default 0.02)
- m_order  : HODC order (default 6)

Outputs:
- dos_vec        : Vector of DOS values
- ldos_mpo_list : List of MPOs representing LDOS at each energy
"""
function compute_dos_ldos_hodc(
    N,
    tn_lis,
    en_num;
    eta = 0.02,
    m_order = 6
)
    @assert length(tn_lis) == N

    # --------------------------------------------------
    # Step 1: raw Chebyshev moments (scalars)
    # --------------------------------------------------
    mus_raw = get_mus_raw(tn_lis)

    # --------------------------------------------------
    # Step 2: energy grid
    # --------------------------------------------------
    yvals = range(-0.99, 0.99; length=en_num)

    # --------------------------------------------------
    # Step 3: precompute HODC parameters
    # --------------------------------------------------
    zl, wl = compute_hodc_params(m_order)

    dos_vec = zeros(Float64, en_num)
    ldos_mpo_list =  []

    # --------------------------------------------------
    # Step 4: main energy loop
    # --------------------------------------------------
    for (i, y) in enumerate(yvals)
        # HODC Chebyshev weights
        nu_k = get_hodc_weights(y, N, eta, zl, wl)

        # ---------- DOS (scalar) ----------
        val = mus_raw[1] * nu_k[1]
        for k in 2:N
            val += 2.0 * mus_raw[k] * nu_k[k]
        end
        dos_vec[i] = val

        # ---------- LDOS MPO ----------
        mpo = nu_k[1] * tn_lis[1]
        for k in 2:N
            mpo = +(mpo, (2.0 * nu_k[k]) * tn_lis[k];maxdim= maxdim)
        end
        ldos_mpo_list[i] = mpo
    end

    # --------------------------------------------------
    # Step 5: normalize DOS 
    # --------------------------------------------------
    dos_vec ./= maximum(dos_vec)
    #filter the negative values from HODC kernel
    dos_vec = max.(0, dos_vec)

    return dos_vec, ldos_mpo_list
end
################################################################################


################################################################################
#the new function replacing _asdiagonal
function mps_to_diagonal_mpo(mps,sites)
    N = length(mps) 
    mpo_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        mps_t = mps[i]
        local old_s
        if i == 1
            old_s = uniqueind(mps_t, mps[i+1])
        elseif i == N
            old_s = uniqueind(mps_t, mps[i-1])
        else
            old_s = uniqueind(mps_t, mps[i-1], mps[i+1])
        end
        s = sites[i]      
        s_p = s'          
        s_temp = Index(dim(s), "temp")
        mpo_tensors[i] = replaceind(mps_t, old_s => s_temp) * delta(s_temp, s, s_p)
    end
    
    return MPO(mpo_tensors)
end
################################################################################


################################################################################
#general order change function
function interleave_mpo(target_mpo, phys_sites, n)
    N_old = length(target_mpo)
    N_new = 2 * N_old
    @assert length(phys_sites) == N_new
    
    new_mpo = MPO(phys_sites)
    
    # 1. Build a mapping table: old Link indices -> arrays of new Link indices
    # Each old Link is split into two new Links, used to bridge across the inserted Identity
    link_map = Dict{Index, Vector{Index}}()
    for k in 1:N_old-1
        ol = linkind(target_mpo, k)
        d = dim(ol)
        # For each old Link, create two new Links with identical dimensions
        link_map[ol] = [Index(d, "Link,l=$(2k-1)"), Index(d, "Link,l=$(2k)")]
    end

    for i in 1:N_old
        idx_orig  = (n == 1) ? 2i-1 : 2i
        idx_ident = (n == 1) ? 2i   : 2i-1
        
        # --- A. Process the operator tensor W ---
        W = target_mpo[i]
        # Replace physical indices
        W = replaceinds(W, siteinds(target_mpo, i) => (phys_sites[idx_orig]', phys_sites[idx_orig]))
        
        # Precisely replace the left Link
        if i > 1
            ol_left = linkind(target_mpo, i-1)
            # The left side of W[i] connects to the *second* new Link in the mapped pair
            W = replaceind(W, ol_left => link_map[ol_left][2])
        end
        
        # Precisely replace the right Link
        if i < N_old
            ol_right = linkind(target_mpo, i)
            # The right side of W[i] connects to the *first* new Link in the mapped pair
            W = replaceind(W, ol_right => link_map[ol_right][1])
        end
        new_mpo[idx_orig] = W
        
        # --- B. Process the identity tensor ID ---
        if idx_ident == 1 || idx_ident == N_new
            # Boundary identities do not carry Link indices
            new_mpo[idx_ident] = delta(phys_sites[idx_ident]', phys_sites[idx_ident])
        else
            # Intermediate identities are sandwiched between two operator tensors
            # They must connect the two new Links originating from the same old Link

            # If n = 1, ID sits at positions 2,4,... between W[i] and W[i+1], corresponding to old_links[i]
            # If n = 0, ID sits at positions 3,5,... between W[i] and W[i+1], corresponding to old_links[i]
            # whether n = 1/0 correspond to x/y or h/e depends on the basis you want, which further relates to diagonal terms

            # Regardless of n, the i-th non-boundary ID always corresponds
            # to the interior of the i-th old Link
            ol = linkind(target_mpo, (idx_ident ÷ 2))
            
            l_left  = link_map[ol][1]
            l_right = link_map[ol][2]
            
            ID = delta(l_left, l_right) * delta(phys_sites[idx_ident]', phys_sites[idx_ident])
            new_mpo[idx_ident] = ID
        end
    end
    #it only changes an MPO to interleaved order, for total MPO, do the summation yourself
    return new_mpo
end
################################################################################


################################################################################
#functions one uses to get MPO for dense matrix
function merge_mps_to_mpo(mps)
    # Length of the original MPS (assumed to be even)
    N = length(mps)
 
    # The resulting MPO has half as many sites
    new_N = N ÷ 2
    mpo = MPO(new_N)
    
    for i in 1:new_N
 
        # Indices of the two neighboring MPS tensors to be merged
        idx1 = 2i - 1
        idx2 = 2i
        
        # Extract the two MPS tensors
        A = mps[idx1]
        B = mps[idx2]
        
        # Contract the two tensors into a single MPO tensor
        # (bond indices are implicitly contracted)
        C = A * B   

        # Assign the merged tensor to the MPO
        mpo[i] = C
    end
    
    return mpo
end

function convert_mpo(old_mps, new_sites)
    # Number of sites in the target MPO
    N = length(new_sites)
  
    # First merge the doubled MPS into an intermediate MPO
    old_mpo = merge_mps_to_mpo(old_mps)
    
    # Initialize the new MPO with the desired physical sites
    new_mpo = MPO(N)
    
    for i in 1:N
        # Original MPS site indices corresponding to this MPO site
        old_i1 = 2i - 1
        old_i2 = 2i
        
        # Physical site indices of the original MPS tensors
        old_s1 = siteind(old_mps, old_i1)
        old_s2 = siteind(old_mps, old_i2)
        
        # New physical input/output indices for the MPO
        new_s_in  = new_sites[i]'   # bra index
        new_s_out = new_sites[i]    # ket index
        
        # Replace the two original physical indices by the new MPO indices
        combined_T = replaceinds(
            old_mpo[i],
            [old_s1, old_s2] => [new_s_in, new_s_out]
        )
        
        # Assign the converted tensor to the new MPO
        new_mpo[i] = combined_T
    end
    
    return new_mpo
end

################################################################################


################################################################################
#direct extraction of the diagonal elements of MPO as MPS, without using QTCI
function extract_diagonal_to_mps(M::MPO)::MPS
    N = length(M)
    new_tensors = Vector{ITensor}(undef, N)

    for i in 1:N
        t = M[i]

        # Get the physical indices for this site (bra, ket)
        si_pair = siteinds(M, i)
        s2 = si_pair[1]   # bra index
        s1 = si_pair[2]   # ket index

        # Dimension of the physical index
        dim_s = dim(s1)

        # Collect all virtual (link) indices, leaving out the physical ones
        v_inds = uniqueinds(t, s1, s2)

        # Create a new MPS tensor with the same virtual indices and one physical index
        res = ITensor(v_inds..., s1)

        # Loop over physical states to extract diagonal elements
        for v in 1:dim_s
            # Take the slice corresponding to s1 = s2 = v
            slice = t * onehot(s1 => v) * onehot(s2 => v)

            # Add this slice back into the resulting MPS tensor at position v
            res += slice * onehot(s1 => v)
        end

        # Store the resulting MPS tensor
        new_tensors[i] = res
    end

    # Build the MPS from the list of site tensors
    return MPS(new_tensors)
end


end # module extra_util