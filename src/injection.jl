# Teleseismic plane waves injected along a box (total-field / scattered-field method): inside the box the solver computes the
# total field, outside only the field scattered by the structure in the box, which the absorbing boundaries take out. The
# incoming field of the layered background (the plane wave with its reverberations and free-surface reflections) is computed
# exactly in the frequency domain (global matrix of the up- and down-going waves in every layer); the finite-difference
# stencils that cross the box boundary are corrected with it every step. The corrections act as sources that do not depend on
# the model inside the box: the adjoint gradients stay exact there.

"""
    TeleseismicWave(wave; angle, box, backazimuth=270.0, t0=0.0, trise, amplitude=1e-6)

A plane `:p`, `:sv` or `:sh` wave from below, injected along the sides and the bottom of `box` (total-field / scattered-field
method): inside the box the simulation gives the total field (the wave, its reflections at the free surface and at layer
interfaces, and the waves scattered by the structure in the box), outside only the scattered waves, which the absorbing
boundaries take out. Use it in `sources`, also for `misfit_gradient` (the gradient is exact inside the box).

- `angle` [°]: angle of incidence in the half-space below the model (from the vertical);
- `box`: `(xmin, xmax, zmin)` in 2D, `(xmin, xmax, ymin, ymax, zmin)` in 3D [km]; the box reaches up to the free surface;
- `backazimuth` [° clockwise from north (y)]: direction towards the source; the wave travels the opposite way (default 270°:
  from the west, towards +x; in 2D only 90° and 270° make sense);
- `t0` [s]: time at which the wave reaches the box first (its deepest corner on the source side);
- `trise` [s] and the source time function `stf` of the simulation: the particle velocity of the incoming wave is
  `amplitude * stf(t)`; for source time functions with integral 1 (the default `:kupper`, `:gaussian`) the wave carries the
  displacement `amplitude` [m].

The model must be horizontally layered (elastic, no water) along the box boundary and below: that is the background in which
the wave is computed exactly (on curvilinear grids it may be VTI or TTI: the incoming quasi-P, quasi-SV or quasi-SH wave,
`angle` that of its phase direction in the half-space). The box must keep 4 cells from the absorbing layers. On curvilinear grids the box is made of
the nodes inside it (and the mesh row nearest to its bottom); the topography must be flat along its sides, and its bottom
5 nodes away from block interfaces.
"""
struct TeleseismicWave
    wave::Symbol
    angle::Float64
    box::Vector{Float64}
    backazimuth::Float64
    t0::Float64
    trise::Float64
    amplitude::Float64
    function TeleseismicWave(wave::Symbol; angle, box, backazimuth=270.0, t0=0.0, trise, amplitude=1e-6)
        wave in (:p, :sv, :sh) || error("TeleseismicWave: wave must be :p, :sv or :sh, got :$wave")
        0 <= angle < 90 || error("TeleseismicWave: angle must be between 0 and 90°")
        length(box) in (3, 5) || error("TeleseismicWave: box = (xmin, xmax, zmin) in 2D or (xmin, xmax, ymin, ymax, zmin) in 3D")
        trise > 0 || error("TeleseismicWave: trise must be positive")
        return new(wave, angle, collect(Float64, box), backazimuth, t0, trise, amplitude)
    end
end

function Base.show(io::IO, w::TeleseismicWave)
    @printf(io, "TeleseismicWave(:%s) at %g° incidence from backazimuth %g°, box %s", w.wave, w.angle, w.backazimuth, w.box)
end

# ------------------------------------------------------------------ layered background, exact plane-wave field

"Layers of the background (from the free surface down, the last one the half-space): depth of the top [km], ρ, λ, μ."
struct Layers
    top::Vector{Float64}
    rho::Vector{Float64}
    lam::Vector{Float64}
    mu::Vector{Float64}
end

"Vertical slowness of the velocity `v` for the horizontal slowness `p` (Im ≤ 0: decaying for the time dependence e^{iωt})."
vslow(v, p) = conj(sqrt(complex(1 / v^2 - p^2)))

"""
Waves of a layer for the horizontal slowness `p`: vertical slownesses and the vectors (u_x, u_z, σ_xz, σ_zz, σ_xx) of unit
amplitude (stresses divided by -iω), for P↓, S↓, P↑, S↑ (P-SV, also σ_yy normal to the plane) or S↓, S↑ with (u_y, σ_yz,
σ_xy) (SH). x is the direction of propagation, z the depth (down).
"""
function layer_waves(ρ, λ, μ, p, ::Val{:psv})
    α, β = sqrt((λ + 2μ) / ρ), sqrt(μ / ρ)
    ηα, ηβ = vslow(α, p), vslow(β, p)
    function b(sz, d)                         # displacement d of a wave with slowness (p, sz)
        div = p * d[1] + sz * d[2]
        return (d[1], d[2], μ * (sz * d[1] + p * d[2]), λ * div + 2μ * sz * d[2], λ * div + 2μ * p * d[1], λ * div)
    end
    sz = (ηα, ηβ, -ηα, -ηβ)
    return sz, (b(ηα, (α * p, α * ηα)), b(ηβ, (β * ηβ, -β * p)), b(-ηα, (α * p, -α * ηα)), b(-ηβ, (-β * ηβ, -β * p)))
end
function layer_waves(ρ, λ, μ, p, ::Val{:sh})
    β = sqrt(μ / ρ)
    ηβ = vslow(β, p)
    b(sz) = (1.0 + 0im, μ * sz, μ * p)
    return (ηβ, -ηβ), (b(ηβ), b(-ηβ))
end

"""
Amplitudes of the waves of every layer for the incoming wave (`:p`, `:sv` or `:sh`, unit displacement, in the half-space at
its top) at the angular frequency `ω`: global matrix of the free surface and the interfaces (continuity of displacement and
traction). Returns a vector per layer (down-going waves referred to the top of the layer, up-going ones to its bottom; in the
half-space both to its top).
"""
function layer_amplitudes(L::Layers, p, wave, ω, ::Val{m}) where {m}
    n = length(L.top)
    nw = m === :psv ? 4 : 2                               # waves per layer
    nb = nw ÷ 2                                           # boundary conditions per interface: nw (continuity), nb (surface)
    W = [layer_waves(L.rho[j], L.lam[j], L.mu[j], p, Val(m)) for j in 1:n]
    h = [L.top[j+1] - L.top[j] for j in 1:n-1]
    # phase of wave w of layer j at its top or bottom: down-going waves are referred to the top, up-going ones to the bottom
    ph(j, w, at_top) = exp(-im * ω * W[j][1][w] * ((at_top ? 0.0 : h[j]) - (w <= nb ? 0.0 : h[j])))
    nu = nw * (n - 1) + nb                                # unknowns: all waves of the layers, the down-going ones of the half-space
    col(j, w) = j < n ? nw * (j - 1) + w : nw * (n - 1) + w
    rows, cols, vals = Int[], Int[], ComplexF64[]
    rhs = zeros(ComplexF64, nu)
    inc = m === :sh ? 2 : wave === :p ? 3 : 4             # the incoming (up-going) wave of the half-space
    traction = m === :psv ? (3, 4) : (2,)
    cont = m === :psv ? (1, 2, 3, 4) : (1, 2)
    function add!(r, j, w, at_top, comp, sgn)
        v = sgn * W[j][2][w][comp] * (j == n ? one(ComplexF64) : ph(j, w, at_top))
        if j == n && w == inc
            rhs[r] -= v
        elseif !(j == n && w > nb)
            push!(rows, r); push!(cols, col(j, w)); push!(vals, v)
        end
    end
    for (q, comp) in enumerate(traction), w in 1:nw          # free surface: no traction at the top of layer 1
        add!(q, 1, w, true, comp, 1)
    end
    for j in 1:n-1, (q, comp) in enumerate(cont), w in 1:nw  # interface below layer j
        r = nb + nw * (j - 1) + q
        add!(r, j, w, false, comp, 1)
        add!(r, j + 1, w, true, comp, -1)
    end
    A = sparse(rows, cols, vals, nu, nu)
    x = A \ rhs
    amp = [x[nw*(j-1)+1:nw*j] for j in 1:n-1]
    push!(amp, [x[nw*(n-1)+1:end]; (w == inc ? 1.0 + 0im : 0.0im for w in nb+1:nw)...])
    return amp, W
end

"Field components (as `layer_waves`) at the depth `z` (below the surface; in the air 0) from the amplitudes `amp`."
function layer_field(L::Layers, amp, W, ω, z, ::Val{m}) where {m}
    nw = m === :psv ? 4 : 2
    nb = nw ÷ 2
    nc = m === :psv ? 6 : 3
    z < 0 && return ntuple(_ -> 0.0im, nc)
    j = searchsortedlast(L.top, z)
    n = length(L.top)
    out = ntuple(_ -> 0.0im, nc)
    for w in 1:nw
        sz = W[j][1][w]
        ref = w <= nb || j == n ? L.top[j] : L.top[j+1]   # down-going (and the half-space): its top; up-going: the bottom
        e = exp(-im * ω * sz * (z - ref))
        out = out .+ amp[j][w] * e .* W[j][2][w]
    end
    return out
end

# ------------------------------------------------------------------ the box and the background profile

"""
Box of the injection on the grid (padded indices): the cells `ic[a]` along the horizontal axes and down to the cell `kB`,
with boundaries a quarter cell beyond the cell centres (no grid point lies on them).
"""
function injection_box(w::TeleseismicWave, g::Grid{N}, na) where {N}
    length(w.box) == (N == 3 ? 5 : 3) || error("TeleseismicWave: box needs $(N == 3 ? 5 : 3) values in $(N)D")
    lims = N == 3 ? ((w.box[1], w.box[2]), (w.box[3], w.box[4])) : ((w.box[1], w.box[2]),)
    ic = ntuple(a -> (cell_index(g, a, lims[a][1]), cell_index(g, a, lims[a][2])), Val(N - 1))
    kB = cell_index(g, N, -w.box[end])
    for a in 1:N-1
        lo, hi = ic[a]
        lo < hi || error("TeleseismicWave: empty box along axis $a")
        (lo - H > na + 4 && hi - H <= g.n[a] - na - 4) ||
            error("TeleseismicWave: the box must keep 4 cells from the absorbing layers (na = $na) at the sides")
    end
    kB - H <= g.n[N] - na - 4 || error("TeleseismicWave: the box must end 4 cells above the absorbing layer at the bottom")
    xc(a, i) = g.beg[a] + (i - H - 0.5) * g.d[a]
    lo = ntuple(a -> xc(a, ic[a][1]) - g.d[a] / 4, Val(N - 1))
    hi = ntuple(a -> xc(a, ic[a][2]) + g.d[a] / 4, Val(N - 1))
    zB = g.zc[kB] + (g.zf[kB] - g.zc[kB]) / 2
    return (; ic, kB, lo, hi, zB)
end

"""
The horizontally layered background along the box: layers of the cells of a column next to the box (equal values merged),
checked against all cells within 4 cells of the box boundary. Depths are measured from the free surface (`zs`, the top of the
first solid cell).
"""
function background_layers(med, g::Grid{N}, box) where {N}
    Iref = ntuple(a -> box.ic[a][1] - 4, N - 1)                   # a column outside the box, on its lower side
    col(f) = [f[Iref..., k] for k in axes(f, N)]
    ρ, λ, μ = col(med.rho), col(med.lam), col(med.mu)
    nz = g.n[N]
    near(a, i) = abs(i - box.ic[a][1]) <= 4 || abs(i - box.ic[a][2]) <= 4
    for I in CartesianIndices(ntuple(a -> box.ic[a][1]-4:box.ic[a][2]+4, N - 1)), k in H+1:H+nz
        (any(a -> near(a, I[a]), 1:N-1) || abs(k - box.kB) <= 4) || continue
        J = (Tuple(I)..., k)
        (med.rho[J...] ≈ ρ[k] && med.lam[J...] ≈ λ[k] && med.mu[J...] ≈ μ[k]) ||
            error("TeleseismicWave: the model must be horizontally layered along the box boundary (it differs at the cell $(J .- H))")
        iszero(med.taup[J...]) && iszero(med.taus[J...]) || error("TeleseismicWave: the model must be elastic (no Qp, Qs) along the box boundary")
    end
    ks = findfirst(k -> μ[k] > 0, H+1:H+nz)
    ks === nothing && error("TeleseismicWave: no solid rock below the box")
    ks += H
    all(k -> μ[k] > 0, ks:H+nz) || error("TeleseismicWave: no water or air below the surface of the background (vs > 0)")
    any(k -> λ[k] > 0 || μ[k] > 0, H+1:ks-1) && error("TeleseismicWave: water is not supported in the background")
    zs = g.zf[ks-1]
    tops, r, l, m = Float64[], Float64[], Float64[], Float64[]
    for k in ks:H+nz
        if isempty(r) || !(ρ[k] ≈ r[end] && λ[k] ≈ l[end] && μ[k] ≈ m[end])
            push!(tops, g.zf[k-1] - zs); push!(r, ρ[k]); push!(l, λ[k]); push!(m, μ[k])
        end
    end
    return Layers(tops, r, l, m), zs
end

"Travel time of the incoming wave (vertical slowness `η` per layer, real part) from the top of the half-space up to the depth `z`."
function vertical_delay(L, η, z)
    zh = L.top[end]
    z >= zh && return -η[end] * (z - zh)
    return sum(j -> η[j] * max(min(L.top[j+1], zh) - max(L.top[j], z), 0.0), 1:length(L.top)-1)
end

# ------------------------------------------------------------------ tables of the incoming field

"Components of the fields of a mode with their staggering (cell face in x, [y,] z)."
injection_components(::Val{:psv}) = (Vx=(true, false), Vz=(false, true), Sxx=(false, false), Szz=(false, false), Sxz=(true, true))
injection_components(::Val{:sh}) = (Vy=(false, false), Syz=(false, true), Sxy=(true, false))
injection_components(::Val{:p3d}) = (Vx=(true, false, false), Vy=(false, true, false), Vz=(false, false, true),
                                     Sxx=(false, false, false), Syy=(false, false, false), Szz=(false, false, false),
                                     Syz=(false, true, true), Sxz=(true, false, true), Sxy=(true, true, false))

"""
The fields of the grid as linear combinations `(component, factor)` of those of the solution in the plane of propagation
(P-SV: u_r, u_z, σ_rz, σ_zz, σ_rr, σ_θθ; SH: u_θ, σ_θz, σ_rθ), for the horizontal direction of propagation `e` and the
transverse direction `θ` (90° counterclockwise from `e` seen from above).
"""
function rotation(::Val{:psv}, e, θ)
    (ex, ey), (tx, ty) = e, θ
    return (Vx=((1, ex),), Vy=((1, ey),), Vz=((2, 1.0),), Sxx=((5, ex^2), (6, tx^2)), Syy=((5, ey^2), (6, ty^2)), Szz=((4, 1.0),),
            Syz=((3, ey),), Sxz=((3, ex),), Sxy=((5, ex * ey), (6, tx * ty)))
end
function rotation(::Val{:sh}, e, θ)
    (ex, ey), (tx, ty) = e, θ
    return (Vx=((1, tx),), Vy=((1, ty),), Vz=(), Sxx=((3, 2ex * tx),), Syy=((3, 2ey * ty),), Szz=(), Syz=((2, ty),),
            Sxz=((2, tx),), Sxy=((3, ex * ty + ey * tx),))
end

"""
Time series (columns: depths) from the spectra `G[q, depth]` at the frequencies `keep[q]` of a real FFT of length `N` (the
others are zero): the first `nτ` samples, divided by `scale`. (Only the frequencies of the source are stored: a full
spectrum per depth needs tens of GB for long simulations.)
"""
function time_table(G::Matrix{ComplexF64}, keep, N, nτ, scale, ::Type{T})::Matrix{T} where {T}
    A = Matrix{T}(undef, nτ, size(G, 2))
    spec = zeros(ComplexF64, N ÷ 2 + 1)
    buf = Vector{Float64}(undef, N)
    P = plan_irfft(spec, N)                               # (one plan and buffer: no GB of garbage from long tables)
    for k in axes(G, 2)
        fill!(spec, 0)                                    # (a c2r transform may overwrite its input)
        spec[keep] .= view(G, :, k)
        LinearAlgebra.mul!(buf, P, spec)
        A[:, k] .= T.(view(buf, 1:nτ) ./ scale)
    end
    return A
end

"""
The frequencies (indices into the spectrum `S` at the angular frequencies `ω`) of the injection tables: those where the
source spectrum exceeds 1e-9 of its maximum, up to `fcap` [Hz]. (A sampled source, linearly interpolated, keeps ~1e-5 of
its peak up to the Nyquist frequency of its sampling: thousands of frequencies that the grid cannot carry.)
"""
table_band(S, ω, fcap) = (m = maximum(abs, S); findall(q -> abs(S[q]) > 1e-9m && ω[q] <= 2π * fcap, eachindex(S)))

"""
The highest frequency [Hz] that a grid with the node spacing `h` carries for the slowest wave of the layers `L` (shear,
or P in fluid layers): its spatial Nyquist frequency v/(2h) (the units of `L` and `h` must match). Higher frequencies
of the source only inject grid noise (fewer than 2 nodes per wavelength).
"""
function grid_fcap(L, h)
    v = minimum(j -> L.mu[j] > 0 ? sqrt(L.mu[j] / L.rho[j]) : sqrt(L.lam[j] / L.rho[j]), eachindex(L.rho))
    return v / 2h
end

"""
The time step of the injection tables: `dt/4`, or coarser for a band-limited source, 1/(100 fmax) with `fmax` the highest
frequency of the tables (`table_band`), when that is larger (linear interpolation in time: relative error ≤ (2π f dτ)²/8,
≤ 5e-4 at fmax). Long simulations otherwise need GB-sized tables.
"""
function table_step(w::TeleseismicWave, stf, τ0, τ1, dt; fcap=Inf)
    dτ = dt / 4
    N = nextpow(2, ceil(Int, 2(τ1 - τ0) / dτ))
    S = rfft([w.amplitude * moment_rate(stf, t, 0.0, w.trise) for t in τ0 .+ (0:N-1) .* dτ])
    keep = table_band(S, 2π .* (0:N÷2) ./ (N * dτ), fcap)
    fmax = isempty(keep) ? 0.0 : (keep[end] - 1) / (N * dτ)
    return fmax > 0 ? max(dτ, 1 / (100fmax)) : dτ
end

"""
Tables `tab[name][it, k]` of the incoming field of the grid components of `mode` at the depths of their points (padded index
`k`, rows 1:kmax) at the times `τ0 + (it - 1) dτ` [internal units: divided by `scale`]. `rot`: the combinations of the
components of the solution in the plane of propagation (`fk`: P-SV or SH).
"""
function injection_tables(w::TeleseismicWave, ::Val{mode}, fk::Val{m}, rot, p, L::Layers, zs, g::Grid, kmax, τ0, τ1, dτ, stf,
                          scale, ::Type{T}) where {mode,m,T}
    wave = m === :sh ? :sh : w.wave
    N = nextpow(2, ceil(Int, 2(τ1 - τ0) / dτ))               # twice the span: no wrap-around of late reverberations
    S = rfft([w.amplitude * moment_rate(stf, t, 0.0, w.trise) for t in τ0 .+ (0:N-1) .* dτ])
    ω = 2π .* (0:N÷2) ./ (N * dτ)
    keep = table_band(S, ω, grid_fcap(L, minimum(g.d)))     # (L, g.d: km and km/s)
    comps = injection_components(Val(mode))
    names = keys(comps)
    depth(k, face) = (face ? g.zf[k] : g.zc[k]) - zs
    G = Dict(n => zeros(ComplexF64, length(keep), kmax) for n in names)
    for (iq, q) in enumerate(keep)
        amp, W = layer_amplitudes(L, p, wave, ω[q], fk)
        for n in names, k in 1:kmax
            f = layer_field(L, amp, W, ω[q], depth(k, comps[n][end]), fk)
            v = sum((f[c] * a for (c, a) in rot[n]); init=0.0im)
            G[n][iq, k] = first(string(n)) == 'V' ? S[q] * v : -S[q] * v         # velocities: G S; stresses: -σ̃ S
        end
    end
    nτ = ceil(Int, (τ1 - τ0) / dτ) + 2
    return NamedTuple{names}(map(n -> time_table(G[n], keep, N, nτ, scale, T), names))
end

# ------------------------------------------------------------------ setup and correction kernels

"""
Setup of the injection: device tables and geometry `W` (for the kernels) and the strips of cells (local `(lo, ndr)`) whose
updates are corrected. `t0` of the wave is the time at which it reaches the box first.
"""
function injection_setup(w::TeleseismicWave, ::Val{mode}, med, g::Grid{N}, na, off, nl, dev, stf, scale, tbeg, dt, nt, ::Type{T}) where {mode,N,T}
    mode === :sh && w.wave !== :sh && error("TeleseismicWave: SH simulations need wave = :sh")
    mode === :psv && w.wave === :sh && error("TeleseismicWave: P-SV simulations need wave = :p or :sv")
    box = injection_box(w, g, na)
    L, zs = background_layers(med, g, box)
    kmax = box.kB + 6
    dτ = dt / 4
    # horizontal direction of propagation and the transverse direction (SH polarization)
    N == 2 && abs(sind(w.backazimuth)) < 0.99 && error("TeleseismicWave: in 2D the backazimuth must be 90° or 270°")
    e3 = N == 2 ? (sign(-sind(w.backazimuth)), 0.0) : (-sind(w.backazimuth), -cosd(w.backazimuth))
    θ3 = (-e3[2], e3[1])
    vel(j) = w.wave === :p ? sqrt((L.lam[j] + 2L.mu[j]) / L.rho[j]) : sqrt(L.mu[j] / L.rho[j])
    p = sind(w.angle) / vel(length(L.top))
    pv = ntuple(a -> p * e3[a], Val(N - 1))                   # horizontal slowness vector
    # τ = t - tshift - pv⋅x: the incoming wave reaches the box first (its deepest corner on the source side) at t0
    ηs = [real(vslow(vel(j), p)) for j in eachindex(L.top)]
    D = vertical_delay(L, ηs, box.zB - zs)
    x_up = ntuple(a -> pv[a] >= 0 ? box.lo[a] : box.hi[a], Val(N - 1))
    tshift = w.t0 - sum(pv .* x_up) - D
    ext = ntuple(a -> (g.beg[a] - H * g.d[a], g.beg[a] + (g.n[a] + H) * g.d[a]), Val(N - 1))   # the model, ghost cells included
    pmin = sum(ntuple(a -> min(pv[a] * ext[a][1], pv[a] * ext[a][2]), Val(N - 1)))
    pmax = sum(ntuple(a -> max(pv[a] * ext[a][1], pv[a] * ext[a][2]), Val(N - 1)))
    τ0 = min(tbeg - tshift - pmax - 2dt, -(vertical_delay(L, ηs, 0.0) - vertical_delay(L, ηs, g.zf[kmax] - zs)) - 2w.trise)
    τ1 = tbeg + (nt + 1) * dt - tshift - pmin + 2dt
    e = N == 2 ? (e3[1], 0.0) : e3
    θ = N == 2 ? (0.0, e3[1]) : θ3
    fk = w.wave === :sh ? Val(:sh) : Val(:psv)
    rot = rotation(fk, e, θ)
    tab = injection_tables(w, Val(mode), fk, rot, p, L, zs, g, kmax, τ0, τ1, dτ, stf, scale, T)
    W = (; tab=map(dev, tab), τ0=T(τ0), dτ=T(dτ), p=T.(pv), tshift=T(tshift),
         x0=ntuple(a -> T(g.beg[a] + (off[a] - H - 0.5) * g.d[a]), Val(N - 1)), dx=ntuple(a -> T(g.d[a]), Val(N - 1)),
         lo=T.(box.lo), hi=T.(box.hi), zB=T(box.zB), zc=dev(T.(g.zc)), zf=dev(T.(g.zf)), kmax=Int32(kmax))
    # strips (global physical cells) around the box boundary, not overlapping: the sides (in 3D the x sides over the whole
    # width in y, the y sides between them) down to the bottom strip, and the bottom strip
    ib = ntuple(a -> box.ic[a] .- H, Val(N - 1))
    kB = box.kB - H
    if N == 2
        (iL, iR), = ib
        strips = ((iL-3:iL+3, 1:kB-4), (iR-3:iR+3, 1:kB-4), (iL-3:iR+3, kB-3:kB+3))
    else
        (iL, iR), (jL, jR) = ib
        strips = ((iL-3:iL+3, jL-3:jR+3, 1:kB-4), (iR-3:iR+3, jL-3:jR+3, 1:kB-4), (iL+4:iR-4, jL-3:jL+3, 1:kB-4),
                  (iL+4:iR-4, jR-3:jR+3, 1:kB-4), (iL-3:iR+3, jL-3:jR+3, kB-3:kB+3))
    end
    return W, map(s -> local_range(s, off, nl), strips)       # (empty on ranks without them)
end

"Horizontal position of the grid point (staggering `s`) at the local padded index `I`, as a tuple."
Base.@propagate_inbounds hpos(W, s, I::CartesianIndex{D}) where {D} =
    ntuple(a -> W.x0[a] + I[a] * W.dx[a] + (s[a] ? W.dx[a] / 2 : zero(W.dx[a])), Val(D - 1))

"Incoming field of the component `name` (staggering `s`) at the grid point `I` (local padded) at the time `t`."
Base.@propagate_inbounds function incoming(W, ::Val{name}, s, I::CartesianIndex{D}, t) where {name,D}
    k = I[D]
    k > W.kmax && return zero(t)
    h = hpos(W, s, I)
    f = (t - W.tshift - sum(W.p .* h) - W.τ0) / W.dτ
    j = floor(Int, f)
    A = getfield(W.tab, name)
    (j < 0 || j + 2 > size(A, 1)) && return zero(t)
    a = f - j
    return (1 - a) * A[j+1, k] + a * A[j+2, k]
end

"Is the grid point (staggering `s`) outside the box? (1 or 0)"
Base.@propagate_inbounds function outside(W, s, I::CartesianIndex{D}) where {D}
    h = hpos(W, s, I)
    z = s[end] ? W.zf[I[D]] : W.zc[I[D]]
    return ifelse(all(W.lo .<= h .<= W.hi) && z <= W.zB, zero(z), one(z))
end

# incoming field at a stencil point as seen from a point with `out` = outside(target): what crosses the box boundary
Base.@propagate_inbounds cross(W, n::Val, s, I, t, out) = incoming(W, n, s, I, t) * (outside(W, s, I) - out)

# --- P-SV
"Strain rates of the incoming field crossing the box boundary (P-SV) at the cell (i, k): the velocity differences of the stress kernel."
Base.@propagate_inbounds function injection_strain_psv(M, c, W, t, i, k)
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    I, ex, ez = CartesianIndex(i, k), CartesianIndex(1, 0), CartesianIndex(0, 1)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], I), two)
    azf, bzf = fd(coefat(c[3], I), two)
    sx, sz = (true, false), (false, true)
    os, oe = outside(W, (false, false), I), outside(W, (true, true), I)          # Sxx, Szz and Sxz points
    vx(J, o) = cross(W, Val(:Vx), sx, J, t, o)
    vz(J, o) = cross(W, Val(:Vz), sz, J, t, o)
    dxVx = (vx(I, os) - vx(I - ex, os)) * ax - (vx(I + ex, os) - vx(I - 2ex, os)) * bx
    dzVz = (vz(I, os) - vz(I - ez, os)) * azf - (vz(I + ez, os) - vz(I - 2ez, os)) * bzf
    dxVz = (vz(I + ex, oe) - vz(I, oe)) * ax - (vz(I + 2ex, oe) - vz(I - ex, oe)) * bx
    dzVx = (vx(I + ez, oe) - vx(I, oe)) * azc - (vx(I + 2ez, oe) - vx(I - ez, oe)) * bzc
    return dxVx, dzVz, dxVz + dzVx
end
"Corrections of the stress updates (P-SV) at the cell (i, k)."
Base.@propagate_inbounds inject_wave_stress_psv!(F, M, c, vc, dt, W, t, i, k) =
    stress_update_psv!(F, M, vc, dt, i, k, injection_strain_psv(M, c, W, t, i, k)...)
"Corrections of the velocity updates (P-SV) at the cell (i, k): the stress differences of the velocity kernel."
Base.@propagate_inbounds function inject_wave_vel_psv!(F, M, c, dt, W, t, i, k)
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    I, ex, ez = CartesianIndex(i, k), CartesianIndex(1, 0), CartesianIndex(0, 1)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], I), two)
    azf, bzf = fd(coefat(c[3], I), two)
    s0, se = (false, false), (true, true)
    ox, oz = outside(W, (true, false), I), outside(W, (false, true), I)          # Vx and Vz points
    sxx(J, o) = cross(W, Val(:Sxx), s0, J, t, o)
    szz(J, o) = cross(W, Val(:Szz), s0, J, t, o)
    sxz(J, o) = cross(W, Val(:Sxz), se, J, t, o)
    dxSxx = (sxx(I + ex, ox) - sxx(I, ox)) * ax - (sxx(I + 2ex, ox) - sxx(I - ex, ox)) * bx
    dzSxz = (sxz(I, ox) - sxz(I - ez, ox)) * azf - (sxz(I + ez, ox) - sxz(I - 2ez, ox)) * bzf
    dxSxz = (sxz(I, oz) - sxz(I - ex, oz)) * ax - (sxz(I + ex, oz) - sxz(I - 2ex, oz)) * bx
    dzSzz = (szz(I + ez, oz) - szz(I, oz)) * azc - (szz(I + 2ez, oz) - szz(I - ez, oz)) * bzc
    vel_update_psv!(F, M, dt, i, k, dxSxx + dzSxz, dxSxz + dzSzz)
end
"""
Part of the corrections in the model gradient (as `grad_psv!`, elastic: the corrections are part of the forward strain rates
at the box boundary), with the adjoint strains `E` of the step.
"""
Base.@propagate_inbounds function inject_wave_grad_psv!(G, E, M, c, dt, W, t, i, k)
    dxVx, dzVz, vxz = injection_strain_psv(M, c, W, t, i, k)
    I = CartesianIndex(i, k)
    a1, a2, a3 = E[1][I] * dt, E[2][I] * dt, E[3][I] * dt
    div = dxVx + dzVz
    G.lam[I] += (a1 + a2) * div
    G.mu[I] += 2(a1 + a2) * div - 2(a1 * dzVz + a2 * dxVx)
    G.xz[I] += a3 * vxz
end

# --- SH
"Strain rates of the incoming field crossing the box boundary (SH) at the cell (i, k)."
Base.@propagate_inbounds function injection_strain_sh(M, c, W, t, i, k)
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    I, ex, ez = CartesianIndex(i, k), CartesianIndex(1, 0), CartesianIndex(0, 1)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], I), two)
    s0 = (false, false)
    oyz, oxy = outside(W, (false, true), I), outside(W, (true, false), I)        # Syz and Sxy points
    vy(J, o) = cross(W, Val(:Vy), s0, J, t, o)
    dxVy = (vy(I + ex, oxy) - vy(I, oxy)) * ax - (vy(I + 2ex, oxy) - vy(I - ex, oxy)) * bx
    dzVy = (vy(I + ez, oyz) - vy(I, oyz)) * azc - (vy(I + 2ez, oyz) - vy(I - ez, oyz)) * bzc
    return dzVy, dxVy
end
Base.@propagate_inbounds inject_wave_stress_sh!(F, M, c, vc, dt, W, t, i, k) =
    stress_update_sh!(F, M, vc, dt, i, k, injection_strain_sh(M, c, W, t, i, k)...)
Base.@propagate_inbounds function inject_wave_vel_sh!(F, M, c, dt, W, t, i, k)
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    I, ex, ez = CartesianIndex(i, k), CartesianIndex(1, 0), CartesianIndex(0, 1)
    ax, bx = fd(c[1], two)
    azf, bzf = fd(coefat(c[3], I), two)
    o = outside(W, (false, false), I)                                             # Vy points
    syz(J) = cross(W, Val(:Syz), (false, true), J, t, o)
    sxy(J) = cross(W, Val(:Sxy), (true, false), J, t, o)
    dzSyz = (syz(I) - syz(I - ez)) * azf - (syz(I + ez) - syz(I - 2ez)) * bzf
    dxSxy = (sxy(I) - sxy(I - ex)) * ax - (sxy(I + ex) - sxy(I - 2ex)) * bx
    vel_update_sh!(F, M, dt, i, k, dxSxy + dzSyz)
end
Base.@propagate_inbounds function inject_wave_grad_sh!(G, E, M, c, dt, W, t, i, k)
    vyz, vxy = injection_strain_sh(M, c, W, t, i, k)
    I = CartesianIndex(i, k)
    G.yz[I] += E[1][I] * dt * vyz
    G.xy[I] += E[2][I] * dt * vxy
end

# --- 3D
"Strain rates of the incoming field crossing the box boundary (3D) at the cell (i, j, k)."
Base.@propagate_inbounds function injection_strain_3d(M, c, W, t, i, j, k)
    two = second_order(k, M.fs_top[i, j], M.fs_bot[i, j], M.ob_top[i, j], M.ob_bot[i, j])
    I = CartesianIndex(i, j, k)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    ax, bx = fd(c[1], two)
    ay, by = fd(c[2], two)
    azc, bzc = fd(coefat(c[3], I), two)
    azf, bzf = fd(coefat(c[4], I), two)
    o0 = outside(W, (false, false, false), I)                                     # Sxx, Syy, Szz
    oyz, oxz, oxy = outside(W, (false, true, true), I), outside(W, (true, false, true), I), outside(W, (true, true, false), I)
    vx(J, o) = cross(W, Val(:Vx), (true, false, false), J, t, o)
    vy(J, o) = cross(W, Val(:Vy), (false, true, false), J, t, o)
    vz(J, o) = cross(W, Val(:Vz), (false, false, true), J, t, o)
    dxVx = (vx(I, o0) - vx(I - ex, o0)) * ax - (vx(I + ex, o0) - vx(I - 2ex, o0)) * bx
    dyVy = (vy(I, o0) - vy(I - ey, o0)) * ay - (vy(I + ey, o0) - vy(I - 2ey, o0)) * by
    dzVz = (vz(I, o0) - vz(I - ez, o0)) * azf - (vz(I + ez, o0) - vz(I - 2ez, o0)) * bzf
    exy = (vy(I + ex, oxy) - vy(I, oxy)) * ax - (vy(I + 2ex, oxy) - vy(I - ex, oxy)) * bx +
          (vx(I + ey, oxy) - vx(I, oxy)) * ay - (vx(I + 2ey, oxy) - vx(I - ey, oxy)) * by
    exz = (vz(I + ex, oxz) - vz(I, oxz)) * ax - (vz(I + 2ex, oxz) - vz(I - ex, oxz)) * bx +
          (vx(I + ez, oxz) - vx(I, oxz)) * azc - (vx(I + 2ez, oxz) - vx(I - ez, oxz)) * bzc
    eyz = (vz(I + ey, oyz) - vz(I, oyz)) * ay - (vz(I + 2ey, oyz) - vz(I - ey, oyz)) * by +
          (vy(I + ez, oyz) - vy(I, oyz)) * azc - (vy(I + 2ez, oyz) - vy(I - ez, oyz)) * bzc
    return dxVx, dyVy, dzVz, eyz, exz, exy
end
Base.@propagate_inbounds inject_wave_stress_3d!(F, M, c, vc, dt, W, t, i, j, k) =
    stress_update_3d!(F, M, vc, dt, i, j, k, injection_strain_3d(M, c, W, t, i, j, k)...)
Base.@propagate_inbounds function inject_wave_vel_3d!(F, M, c, dt, W, t, i, j, k)
    two = second_order(k, M.fs_top[i, j], M.fs_bot[i, j], M.ob_top[i, j], M.ob_bot[i, j])
    I = CartesianIndex(i, j, k)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    ax, bx = fd(c[1], two)
    ay, by = fd(c[2], two)
    azc, bzc = fd(coefat(c[3], I), two)
    azf, bzf = fd(coefat(c[4], I), two)
    ox, oy, oz = outside(W, (true, false, false), I), outside(W, (false, true, false), I), outside(W, (false, false, true), I)
    S(n, s, J, o) = cross(W, n, s, J, t, o)            # n: Val(name), resolved at compile time
    s0, syz, sxz, sxy = (false, false, false), (false, true, true), (true, false, true), (true, true, false)
    dSx = (S(Val(:Sxx), s0, I + ex, ox) - S(Val(:Sxx), s0, I, ox)) * ax - (S(Val(:Sxx), s0, I + 2ex, ox) - S(Val(:Sxx), s0, I - ex, ox)) * bx +
          (S(Val(:Sxy), sxy, I, ox) - S(Val(:Sxy), sxy, I - ey, ox)) * ay - (S(Val(:Sxy), sxy, I + ey, ox) - S(Val(:Sxy), sxy, I - 2ey, ox)) * by +
          (S(Val(:Sxz), sxz, I, ox) - S(Val(:Sxz), sxz, I - ez, ox)) * azf - (S(Val(:Sxz), sxz, I + ez, ox) - S(Val(:Sxz), sxz, I - 2ez, ox)) * bzf
    dSy = (S(Val(:Sxy), sxy, I, oy) - S(Val(:Sxy), sxy, I - ex, oy)) * ax - (S(Val(:Sxy), sxy, I + ex, oy) - S(Val(:Sxy), sxy, I - 2ex, oy)) * bx +
          (S(Val(:Syy), s0, I + ey, oy) - S(Val(:Syy), s0, I, oy)) * ay - (S(Val(:Syy), s0, I + 2ey, oy) - S(Val(:Syy), s0, I - ey, oy)) * by +
          (S(Val(:Syz), syz, I, oy) - S(Val(:Syz), syz, I - ez, oy)) * azf - (S(Val(:Syz), syz, I + ez, oy) - S(Val(:Syz), syz, I - 2ez, oy)) * bzf
    dSz = (S(Val(:Sxz), sxz, I, oz) - S(Val(:Sxz), sxz, I - ex, oz)) * ax - (S(Val(:Sxz), sxz, I + ex, oz) - S(Val(:Sxz), sxz, I - 2ex, oz)) * bx +
          (S(Val(:Syz), syz, I, oz) - S(Val(:Syz), syz, I - ey, oz)) * ay - (S(Val(:Syz), syz, I + ey, oz) - S(Val(:Syz), syz, I - 2ey, oz)) * by +
          (S(Val(:Szz), s0, I + ez, oz) - S(Val(:Szz), s0, I, oz)) * azc - (S(Val(:Szz), s0, I + 2ez, oz) - S(Val(:Szz), s0, I - ez, oz)) * bzc
    vel_update_3d!(F, M, dt, i, j, k, dSx, dSy, dSz)
end
Base.@propagate_inbounds function inject_wave_grad_3d!(G, E, M, c, dt, W, t, i, j, k)
    dxVx, dyVy, dzVz, vyz, vxz, vxy = injection_strain_3d(M, c, W, t, i, j, k)
    I = CartesianIndex(i, j, k)
    a = ntuple(m -> E[m][I] * dt, Val(6))
    div = dxVx + dyVy + dzVz
    o = (dyVy + dzVz, dxVx + dzVz, dxVx + dyVy)
    G.lam[I] += (a[1] + a[2] + a[3]) * div
    G.mu[I] += 2(a[1] + a[2] + a[3]) * div - 2(a[1] * o[1] + a[2] * o[2] + a[3] * o[3])
    G.yz[I] += a[4] * vyz
    G.xz[I] += a[5] * vxz
    G.xy[I] += a[6] * vxy
end

injection_kernels(::Val{:psv}) = (inject_wave_stress_psv!, inject_wave_vel_psv!, inject_wave_grad_psv!)
injection_kernels(::Val{:sh}) = (inject_wave_stress_sh!, inject_wave_vel_sh!, inject_wave_grad_sh!)
injection_kernels(::Val{:p3d}) = (inject_wave_stress_3d!, inject_wave_vel_3d!, inject_wave_grad_3d!)

"Apply the corrections of the stress (`Val(:stress)`, incoming velocities at `t`) or velocity updates (incoming stresses)."
inject_wave!(::Nothing, args...) = nothing
function inject_wave!(tw::Tuple, ::Val{:stress}, F, M, c, vc, dtT, t, backend)
    W, ranges, (ks, _, _) = tw
    foreach(((lo, ndr),) -> all(>(0), ndr) && foreach_cell!(backend, ks, (F, M, c, vc, dtT, W, t), lo, ndr), ranges)
end
function inject_wave!(tw::Tuple, ::Val{:velocity}, F, M, c, vc, dtT, t, backend)
    W, ranges, (_, kv, _) = tw
    foreach(((lo, ndr),) -> all(>(0), ndr) && foreach_cell!(backend, kv, (F, M, c, dtT, W, t), lo, ndr), ranges)
end
"Add the part of the corrections of the stress updates to the model gradient (adjoint strains `E` of the step at `t`)."
inject_wave_grad!(::Nothing, args...) = nothing
function inject_wave_grad!(tw::Tuple, G, E, M, c, dtT, t, backend)
    W, ranges, (_, _, kg) = tw
    foreach(((lo, ndr),) -> all(>(0), ndr) && foreach_cell!(backend, kg, (G, E, M, c, dtT, W, t), lo, ndr), ranges)
end
