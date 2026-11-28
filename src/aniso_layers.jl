# The incoming field of a TeleseismicWave in a horizontally layered background of VTI or TTI layers (curvilinear grids):
# exact, in the frequency domain, as for isotropic layers (injection.jl), with the waves of every layer from the
# first-order system of the displacement u and the traction t on horizontal planes (Stroh): for the horizontal slowness
# (p₁, p₂) and e^{iω(t - p₁x - p₂y - qz)} (z down), q [u; t̃] = N [u; t̃] with t̃ = t/(-iω),
#     N = [-T⁻¹S  T⁻¹; ρI + SᵀT⁻¹S - W  -SᵀT⁻¹],  T_ik = C_i3k3, S_ik = pα C_i3kα, W_ik = pα pβ C_iαkβ.
# Its six eigenvalues are the vertical slownesses of the three down-going and three up-going waves (by the vertical energy
# flux Re(t̃·ū), or the decay of evanescent waves); a global matrix of the free surface and the interfaces gives their
# amplitudes. The waves are ordered (quasi-P, quasi-SV, quasi-SH) by their polarization; in isotropic layers (equal S
# slownesses) the S waves are SV and SH. Coordinates: x, y of the grid, z down (stiffness reflected).

"Layers of an anisotropic background (from the free surface down, the last one the half-space): top [m], ρ, stiffness (Voigt, x y z-down, SI)."
struct AnisoLayers
    top::Vector{Float64}
    rho::Vector{Float64}
    C::Vector{Matrix{Float64}}
end

voigt(i, j) = i == j ? i : 9 - i - j                  # (2,3) → 4, (1,3) → 5, (1,2) → 6
const VOIGT_PAIRS = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
@inline cijkl(C, i, j, k, l) = C[voigt(i, j), voigt(k, l)]

"""
The stiffness (Voigt, SI) of a transversely isotropic medium in the frame x, y, z down: the VTI moduli of ρ, vp, vs and the
Thomsen parameters (elastic, as `vti_moduli`), rotated to the axis of tilt θ and azimuth φ [rad] (`axis_frame`).
"""
function tti_voigt(ρ, vp, vs, ε, δ, γ, θ, φ)
    c33, c55 = ρ * vp^2, ρ * vs^2
    c11, c66 = c33 * (1 + 2ε), c55 * (1 + 2γ)
    c13 = sqrt(max((c33 - c55) * (c33 * (1 + 2δ) - c55), 0.0)) - c55
    C0 = zeros(6, 6)
    C0[1, 1] = C0[2, 2] = c11; C0[3, 3] = c33; C0[1, 2] = C0[2, 1] = c11 - 2c66
    C0[1, 3] = C0[3, 1] = C0[2, 3] = C0[3, 2] = c13; C0[4, 4] = C0[5, 5] = c55; C0[6, 6] = c66
    A = axis_frame(θ, φ)
    C = zeros(6, 6)
    for I in 1:6, J in I:6
        (i, j), (k, l) = VOIGT_PAIRS[I], VOIGT_PAIRS[J]
        s = 0.0
        for p in 1:3, q in 1:3, r in 1:3, t in 1:3
            s += A[p][i] * A[q][j] * A[r][k] * A[t][l] * cijkl(C0, p, q, r, t)
        end
        C[I, J] = C[J, I] = s
    end
    nz = (0, 0, 2, 1, 1, 0)                            # z down: an odd number of z indices changes the sign
    for I in 1:6, J in 1:6
        isodd(nz[I] + nz[J]) && (C[I, J] = -C[I, J])
    end
    return C
end

"Reference polarizations (P, SV, SH) of a wave with horizontal slowness `p` along the unit vector `e` and vertical slowness `η` (z down)."
function wave_refs(p, η, e)
    P = [p * e[1], p * e[2], η]
    SV = [η * e[1], η * e[2], -p]                       # (isotropic: P (α p, α η), S (β η, -β p) as `layer_waves`)
    return P / LinearAlgebra.norm(P), SV / LinearAlgebra.norm(SV), [-e[2], e[1], 0.0]
end

"""
The waves of an anisotropic layer for the horizontal slowness `pv = (p₁, p₂)` [s/m] (along the unit vector `e`): vertical
slownesses and the vectors (u₁, u₂, u₃, σ̃₁₁, σ̃₂₂, σ̃₃₃, σ̃₂₃, σ̃₁₃, σ̃₁₂) of unit displacement (σ̃ = σ/(-iω)), for the
down-going (P, SV, SH) and the up-going (P, SV, SH) waves.
"""
function aniso_layer_waves(ρ, C, pv, e)
    p1, p2 = pv
    T = [cijkl(C, i, 3, k, 3) for i in 1:3, k in 1:3]
    S = [p1 * cijkl(C, i, 3, k, 1) + p2 * cijkl(C, i, 3, k, 2) for i in 1:3, k in 1:3]
    W = [p1^2 * cijkl(C, i, 1, k, 1) + p1 * p2 * (cijkl(C, i, 1, k, 2) + cijkl(C, i, 2, k, 1)) + p2^2 * cijkl(C, i, 2, k, 2) for i in 1:3, k in 1:3]
    Ti = inv(T)
    N = [-Ti*S Ti; ρ*LinearAlgebra.I(3)+S'*Ti*S-W -S'*Ti]
    F = LinearAlgebra.eigen(N)
    q, V = complex.(F.values), complex.(F.vectors)
    isdown(m) = abs(imag(q[m])) > 1e-6 * abs(q[m]) ? imag(q[m]) < 0 : real(LinearAlgebra.dot(V[1:3, m], V[4:6, m])) > 0
    groups = (filter(isdown, 1:6), filter(!isdown, 1:6))
    all(g -> length(g) == 3, groups) || error("TeleseismicWave: could not separate the up- and down-going waves of a layer")
    pa = hypot(p1, p2)
    qs, vecs = ComplexF64[], Vector{ComplexF64}[]
    for (down, g) in zip((true, false), groups)
        η = sum(m -> abs(real(q[m])), g) / 3
        refs = wave_refs(pa, down ? η : -η, e)
        u(m) = V[1:3, m] / LinearAlgebra.norm(V[1:3, m])
        ov(m, r) = abs(LinearAlgebra.dot(conj(u(m)), refs[r]))
        mP = g[argmax([ov(m, 1) for m in g])]
        rest = filter(!=(mP), g)
        mSV = rest[argmax([ov(m, 2) for m in rest])]
        mSH = only(filter(!=(mSV), rest))
        cols = [V[:, mP], V[:, mSV], V[:, mSH]]
        qq = [q[mP], q[mSV], q[mSH]]
        if abs(qq[2] - qq[3]) <= 1e-8 * maximum(abs, q)    # (degenerate S waves: the SV and SH combinations)
            B = hcat(cols[2], cols[3])
            cols[2], cols[3] = B * (B[1:3, :] \ refs[2]), B * (B[1:3, :] \ refs[3])
            qq[3] = qq[2]
        end
        for (r, (v, qm)) in enumerate(zip(cols, qq))
            uu = v[1:3]
            ph = transpose(uu) * refs[r]                   # (unit displacement, its component along the reference positive)
            v = v * (abs(ph) > 0 ? conj(ph) / abs(ph) : 1.0) / LinearAlgebra.norm(uu)
            s = (p1, p2, qm)
            σ = [sum(cijkl(C, i, j, k, l) * s[l] * v[k] for k in 1:3, l in 1:3) for (i, j) in VOIGT_PAIRS]
            push!(qs, qm); push!(vecs, [v[1:3]; σ])
        end
    end
    return qs, vecs
end

"""
Amplitudes of the waves of every layer for the incoming up-going wave `wave` (`:p`, `:sv`, `:sh`; unit displacement in the
half-space at its top) at the angular frequency `ω`, from the waves `W` of the layers (`aniso_layer_waves`): global matrix of
the free surface and the interfaces, as `layer_amplitudes`.
"""
function aniso_layer_amplitudes(L::AnisoLayers, W, wave, ω)
    n = length(L.top)
    nw, nb = 6, 3
    h = [L.top[j+1] - L.top[j] for j in 1:n-1]
    ph(j, w, at_top) = exp(-im * ω * W[j][1][w] * ((at_top ? 0.0 : h[j]) - (w <= nb ? 0.0 : h[j])))
    nu = nw * (n - 1) + nb
    col(j, w) = j < n ? nw * (j - 1) + w : nw * (n - 1) + w
    rows, cols, vals = Int[], Int[], ComplexF64[]
    rhs = zeros(ComplexF64, nu)
    inc = nb + (wave === :p ? 1 : wave === :sv ? 2 : 3)
    traction, cont = (8, 7, 6), (1, 2, 3, 8, 7, 6)            # σ̃13, σ̃23, σ̃33; and u
    function add!(r, j, w, at_top, comp, sgn)
        v = sgn * W[j][2][w][comp] * (j == n ? one(ComplexF64) : ph(j, w, at_top))
        if j == n && w == inc
            rhs[r] -= v
        elseif !(j == n && w > nb)
            push!(rows, r); push!(cols, col(j, w)); push!(vals, v)
        end
    end
    for (q, comp) in enumerate(traction), w in 1:nw
        add!(q, 1, w, true, comp, 1)
    end
    for j in 1:n-1, (q, comp) in enumerate(cont), w in 1:nw
        r = nb + nw * (j - 1) + q
        add!(r, j, w, false, comp, 1)
        add!(r, j + 1, w, true, comp, -1)
    end
    x = sparse(rows, cols, vals, nu, nu) \ rhs
    amp = [x[nw*(j-1)+1:nw*j] for j in 1:n-1]
    push!(amp, [x[nw*(n-1)+1:end]; (w == inc ? 1.0 + 0im : 0.0im for w in nb+1:nw)...])
    return amp
end

"The field (u₁, u₂, u₃, σ̃₁₁, σ̃₂₂, σ̃₃₃, σ̃₂₃, σ̃₁₃, σ̃₁₂) at the depth `z` (in the air 0), as `layer_field`."
function aniso_layer_field(L::AnisoLayers, amp, W, ω, z)
    out = zeros(ComplexF64, 9)
    z < 0 && return out
    j = searchsortedlast(L.top, z)
    n = length(L.top)
    for w in 1:6
        ref = w <= 3 || j == n ? L.top[j] : L.top[j+1]
        out .+= (amp[j][w] * exp(-im * ω * W[j][1][w] * (z - ref))) .* W[j][2][w]
    end
    return out
end

"""
The horizontal slowness [s/m] of the incoming wave `wave` with the angle of incidence `angle` [°] (of its phase direction,
from the vertical) in the half-space of `L`, travelling along the unit vector `e`: from the Christoffel equation.
"""
function aniso_incidence(L::AnisoLayers, wave, angle, e)
    ρ, C = L.rho[end], L.C[end]
    si, ci = sind(angle), cosd(angle)
    n = (si * e[1], si * e[2], -ci)                       # (up-going, z down)
    Γ = [sum(cijkl(C, i, j, k, l) * n[j] * n[l] for j in 1:3, l in 1:3) for i in 1:3, k in 1:3]
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Γ))
    refs = wave_refs(si, -ci, e)
    r = wave === :p ? 1 : wave === :sv ? 2 : 3
    pick = wave === :p ? argmax(F.values) : begin
        S = filter(!=(argmax(F.values)), 1:3)
        S[argmax([abs(LinearAlgebra.dot(F.vectors[:, m], refs[r])) for m in S])]
    end
    return si / sqrt(F.values[pick] / ρ)
end

"The slowest phase velocity of S waves in the layers (over directions): for the frequency cap of the tables (`grid_fcap`)."
function grid_fcap(L::AnisoLayers, h)
    v = Inf
    for (ρ, C) in zip(L.rho, L.C), θ in 0:10:180, φ in 0:30:330
        n = (sind(θ) * cosd(φ), sind(θ) * sind(φ), cosd(θ))
        Γ = [sum(cijkl(C, i, j, k, l) * n[j] * n[l] for j in 1:3, l in 1:3) for i in 1:3, k in 1:3]
        v = min(v, sqrt(minimum(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Γ))) / ρ))
    end
    return v / 2h
end

"Tables of the incoming field in anisotropic layers (as `teleseismic_tables_cv`; grid components x, y, z down, SI)."
function teleseismic_tables_cv(w, L::AnisoLayers, pv, e, zt, names, τ0, τ1, dτ, stf, ::Type{T}; fcap=Inf) where {T}
    N = nextpow(2, ceil(Int, 2(τ1 - τ0) / dτ))
    S = rfft([w.amplitude * moment_rate(stf, t, 0.0, w.trise) for t in τ0 .+ (0:N-1) .* dτ])
    ω = 2π .* (0:N÷2) ./ (N * dτ)
    keep = table_band(S, ω, fcap)
    W = [aniso_layer_waves(L.rho[j], L.C[j], pv, e) for j in eachindex(L.top)]     # (independent of ω)
    comp = (Vx=1, Vy=2, Vz=3, Sxx=4, Syy=5, Szz=6, Syz=7, Sxz=8, Sxy=9)
    G = Dict(n => zeros(ComplexF64, length(keep), length(zt)) for n in names)
    for (iq, q) in enumerate(keep)
        amp = aniso_layer_amplitudes(L, W, w.wave, ω[q])
        for (kz, z) in enumerate(zt)
            f = aniso_layer_field(L, amp, W, ω[q], z)
            for n in names
                G[n][iq, kz] = first(string(n)) == 'V' ? S[q] * f[comp[n]] : -S[q] * f[comp[n]]
            end
        end
    end
    nτ = ceil(Int, (τ1 - τ0) / dτ) + 2
    return NamedTuple{names}(map(n -> time_table(G[n], keep, N, nτ, 1.0, T), names)), W
end
