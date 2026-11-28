# Material setup on the global grid (host), ported from OpenSWPC `m_medium.f90` / `m_absorb_c.f90`.
#
# Internal layout: arrays are indexed (i, [j,] k) with k = 1 at the TOP of the model (depth increases with k),
# padded by H = 2 ghost cells on every side. Physical cell i lives at array index i + H.

const H = 2

struct Grid{N}
    n::NTuple{N,Int}          # physical cells (x, [y,] z)
    d::NTuple{N,Float64}      # spacing [km]; in depth the mean spacing if the grid is stretched
    beg::NTuple{N,Float64}    # cell-edge start; for z this is the DEPTH of the top edge
    zc::Vector{Float64}       # depths of the cell centres, padded (index k)
    zf::Vector{Float64}       # depths of the faces below them (between centres k and k + 1)
    uniform::Bool             # constant spacing in depth
end
Grid{N}(n, d, beg) where {N} = (z = beg[N] .+ ((1:n[N]+2H) .- H .- 0.5) .* d[N]; Grid{N}(n, d, beg, z, z .+ d[N] / 2, true))

"""
Concretely typed copy of the parts of a `CartData` the solver needs (GMG stores them untyped):
coordinate axes and the fields `rho`, `vp`, `vs`, `Qp`, `Qs` as `Array{Float64,3}` (Q = Inf if missing), and with
`Val(true)` the Thomsen parameters `epsilon`, `delta`, `gamma` (0 if missing).
"""
function model_arrays(d::CartData, ::Val{true})
    m = model_arrays(d)
    field(name) = haskey(d.fields, name) ? convert(Array{Float64,3}, d.fields[name])::Array{Float64,3} : zeros(size(m.rho))
    return merge(m, (; epsilon=field(:epsilon), delta=field(:delta), gamma=field(:gamma)))
end
model_arrays(d::CartData, ::Val{false}) = model_arrays(d)
function model_arrays(d::CartData)
    x, y, z = model_axes(d)
    function field(name)
        haskey(d.fields, name) && return convert(Array{Float64,3}, d.fields[name])::Array{Float64,3}
        name in (:Qp, :Qs) || error("the CartData model needs fields `rho`, `vp` and `vs` (and optionally `Qp`, `Qs`)")
        return fill(Inf, length(x), length(y), length(z))
    end
    return (; x, y, z, rho=field(:rho), vp=field(:vp), vs=field(:vs), Qp=field(:Qp), Qs=field(:Qs))
end

"Coordinates `(; x, y, z)` [km] of the cell centres along the axes of the model (z up)."
function model_axes(d::CartData)
    axis(A, I...) = Vector{Float64}(A[I...])::Vector{Float64}
    return (; x=axis(d.x.val, :, 1, 1), y=axis(d.y.val, 1, :, 1), z=axis(d.z.val, 1, 1, :))
end

"""
    DistributedModel(f, x, y, z; attenuation=true, anisotropy=false)

A model that is built piece by piece, for runs on several MPI processes with models too large to hold as a whole: every
process asks only for the part of the model in its own subdomain (see `nproc` of [`simulate`](@ref)).

`x`, `y`, `z` are the coordinates [km] of the cell centres of the whole grid (z up; constant spacing in x and y, constant or
smoothly varying in z), as for a `CartData` made with `xyz_grid(x, y, z)`; a single `y` gives a 2D model. `f(ix, iy)` returns
the model of the columns `x[ix]`, `y[iy]` (index ranges) over all of `z`: a `CartData`, or a `NamedTuple` of arrays of size
`(length(ix), length(iy), length(z))`, with the fields `rho`, `vp`, `vs` and optionally `Qp`, `Qs` (as in a `CartData` model).
It is called once per process, for a range that reaches a few cells into the neighbouring subdomains and leaves out the
absorbing layers at the sides, whose medium continues the cells next to them.

`attenuation = false` leaves out the memory variables of the attenuation (for models without finite `Qp`, `Qs`).
`anisotropy = true` makes the model VTI: `f` may then also return the Thomsen parameters `epsilon`, `delta`, `gamma` (0 if
missing).

```julia
x, y, z = range(0, 100, 2001), range(0, 80, 1601), range(-30, 2, 641)
model = DistributedModel(x, y, z; attenuation=false) do ix, iy
    X, Y, Z = xyz_grid(x[ix], y[iy], z)
    air = Z .> 0
    (rho=ifelse.(air, 1.0, 2700.0), vp=ifelse.(air, 0.0, 5.0), vs=ifelse.(air, 0.0, 2.9))
end
result = simulate(model; nproc=(2, 2), dt, nt, sources, stations)
```
Sources are `MomentTensorSource`s and `ForceSource`s at their cells (`interpolation = :cell`); plane and teleseismic waves,
`interpolation = :sinc`, [`misfit_gradient`](@ref) and [`born`](@ref) need a `CartData`. The wavefield at the end is not
collected, unless `simulate` is called with `wavefield = true`.

    DistributedModel(f, x, y, nz::Integer; attenuation=true, anisotropy=false)

The same for a curvilinear grid (see [`curvilinear_mesh`](@ref)): `x`, `y` are the nodes [km] of the columns (the SBP nodes,
e.g. `sbp_nodes(a, b, n)` of GMG), `nz` the number of nodes in every column, and `f(ix, iy)` returns the deformed
`CartData` of the columns `x[ix]`, `y[iy]` (size `(length(ix), length(iy), nz)`, e.g. made with GMG's `curvilinear_grid`
and a fixed number of cells per layer) with the fields `rho`, `vp`, `vs` (optionally `Qp`, `Qs`, `epsilon`, `delta`,
`gamma`), or a `NamedTuple` with these and the heights `Z` [km]. Every process asks for its columns and a few more.
"""
struct DistributedModel{F}
    f::F
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    attenuation::Bool
    anisotropy::Bool
    nzc::Int                            # nodes per column of a curvilinear model (0: a regular grid with the cells `z`)
end
DistributedModel(f, x, y, z; attenuation=true, anisotropy=false) =
    DistributedModel(f, vec(collect(Float64, x)), vec(collect(Float64, y)), vec(collect(Float64, z)), attenuation, anisotropy, 0)
DistributedModel(f, x, y, nz::Integer; attenuation=true, anisotropy=false) =
    DistributedModel(f, vec(collect(Float64, x)), vec(collect(Float64, y)), Float64[], attenuation, anisotropy, Int(nz))
is_curvilinear(m::DistributedModel) = m.nzc > 0
Base.size(m::DistributedModel) = (length(m.x), length(m.y), is_curvilinear(m) ? m.nzc : length(m.z))
model_axes(m::DistributedModel) = (; x=m.x, y=m.y, z=m.z)

function Base.show(io::IO, m::DistributedModel)
    print(io, "DistributedModel of ", join(size(m), "×"), is_curvilinear(m) ? " nodes (curvilinear)" : " cells", m.attenuation ? "" : ", no attenuation", m.anisotropy ? ", VTI" : "")
end

"""
Fields `rho` [g/cm³], `vp`, `vs`, `Qp`, `Qs` of the cells `S` (index ranges in x, (y); all depths) as internal arrays: cut
out of the arrays `m` of a whole model.
"""
function model_block(m::NamedTuple, S::NTuple{M,UnitRange{Int}}, ::Val{N}) where {M,N}
    cut(A) = internal(N == 3 ? view(A, S[1], S[2], :) : view(A, S[1], :, :), Val(N))
    b = (; rho=cut(m.rho), vp=cut(m.vp), vs=cut(m.vs), Qp=cut(m.Qp), Qs=cut(m.Qs))
    return haskey(m, :epsilon) ? merge(b, (; epsilon=cut(m.epsilon), delta=cut(m.delta), gamma=cut(m.gamma))) : b
end
"""
The same, built by a [`DistributedModel`](@ref); `rho_max` is the largest density as given (`rho_factor` converts it to g/cm³).
With `Val(true)` also the Thomsen parameters.
"""
function model_block(model::DistributedModel, S::NTuple{M,UnitRange{Int}}, ::Val{N}, rho_factor, ::Val{aniso}=Val(false)) where {M,N,aniso}
    ix, iy = S[1], N == 3 ? S[2] : (1:1)
    b = model.f(ix, iy)
    fields = b isa CartData ? b.fields : b
    sz = (length(ix), length(iy), length(model.z))
    function field(name)
        if haskey(fields, name)
            A = convert(Array{Float64,3}, fields[name])::Array{Float64,3}
            size(A) == sz || error("DistributedModel: the field `$name` of the columns ($ix, $iy) has size $(size(A)), expected $sz")
            return A
        end
        name in (:epsilon, :delta, :gamma) && return zeros(sz)
        name in (:Qp, :Qs) || error("a DistributedModel must return the fields `rho`, `vp` and `vs` (and optionally `Qp`, `Qs`)")
        return fill(Inf, sz)
    end
    any(f -> haskey(fields, f), (:tilt, :azimuth)) && error("a tilted symmetry axis (TTI) needs a curvilinear DistributedModel")
    !aniso && any(f -> haskey(fields, f), (:epsilon, :delta, :gamma)) &&
        error("the DistributedModel returns Thomsen parameters: create it with `anisotropy = true`")
    rho = field(:rho)
    cut(A) = internal(A, Val(N))
    b = (; rho=cut(rho ./ rho_factor), vp=cut(field(:vp)), vs=cut(field(:vs)), Qp=cut(field(:Qp)), Qs=cut(field(:Qs)),
         rho_max=maximum(rho))
    return aniso ? merge(b, (; epsilon=cut(field(:epsilon)), delta=cut(field(:delta)), gamma=cut(field(:gamma)))) : b
end

"""
Part of an array of the padded global grid: the cells `r` (global index ranges) that an MPI rank holds, indexed with global
indices. The medium of a [`DistributedModel`](@ref) consists of these: no rank has the whole.
"""
struct Window{T,N}
    a::Array{T,N}
    r::NTuple{N,UnitRange{Int}}
end
Base.@propagate_inbounds Base.getindex(w::Window{T,N}, I::Vararg{Integer,N}) where {T,N} = w.a[ntuple(d -> I[d] - first(w.r[d]) + 1, Val(N))...]
Base.@propagate_inbounds Base.getindex(w::Window{T,N}, I::CartesianIndex{N}) where {T,N} = w[Tuple(I)...]
Base.@propagate_inbounds Base.getindex(w::Window{T,N}, R::Vararg{AbstractUnitRange{<:Integer},N}) where {T,N} =
    w.a[ntuple(d -> R[d] .- (first(w.r[d]) - 1), Val(N))...]
"Is the cell `I` (global indices) among the cells held on this rank?"
held(w::Window{T,N}, I::Vararg{Integer,N}) where {T,N} = all(ntuple(d -> I[d] in w.r[d], Val(N)))
held(a::Array{T,N}, I::Vararg{Integer,N}) where {T,N} = checkbounds(Bool, a, I...)

"""
Grid of the model: every `CartData` node becomes a cell centre. The spacing is constant in x (and y); in depth it may vary
smoothly (a stretched grid, e.g. fine near the surface): the faces between the cells are then interpolated (4th order) from
the centres.
"""
function Grid(m::NamedTuple, ::Val{N}) where {N}
    axes = N == 3 ? (m.x, m.y, -reverse(m.z)) : (m.x, -reverse(m.z))
    for a in axes[1:N-1]
        length(a) > 1 && all(≈(a[2] - a[1]), diff(a)) || error("CartData must be a regular grid in x (and y), with at least 2 points per axis")
    end
    z = axes[N]
    dz = diff(z)
    length(z) > 1 && all(>(0), dz) || error("CartData needs at least 2 increasing points in z")
    uniform = all(≈(dz[1]), dz)
    d = (map(a -> a[2] - a[1], axes[1:N-1])..., uniform ? dz[1] : (z[end] - z[1]) / (length(z) - 1))   # stretched: the mean
    uniform && return Grid{N}(map(length, axes), d, map((a, h) -> a[1] - h / 2, axes, d))
    r = maximum(i -> max(dz[i+1] / dz[i], dz[i] / dz[i+1]), 1:length(dz)-1)
    r <= 1.25 || error("the spacing in z must change smoothly (at most 25 % from cell to cell, here $(round(100(r - 1)))) %)")
    # padded centres (ghost cells continue the spacing at the ends) and faces (4th-order interpolation; 2nd order at the ends)
    n = length(z)
    zc = [z[1] .- (H:-1:1) .* dz[1]; z; z[end] .+ (1:H) .* dz[end]]
    zf = [k in 2:n+2H-2 ? 9 / 16 * (zc[k] + zc[k+1]) - 1 / 16 * (zc[k-1] + zc[k+2]) : (zc[k] + zc[min(k + 1, end)]) / 2 for k in 1:n+2H]
    zf[end] = zc[end] + dz[end] / 2
    top = zc[H+1] - (zf[H+1] - zc[H+1])
    return Grid{N}(map(length, axes), d, (map((a, h) -> a[1] - h / 2, axes[1:N-1], d[1:N-1])..., top), zc, zf, false)
end

"""
Courant number of the time step `dt`: on a stretched grid per layer, with its largest P velocity and its spacing in depth
(stable below 6/7).
"""
function courant_number(med, g::Grid{N}, dt) where {N}
    g.uniform && return med.vmax * dt * sqrt(sum(1 ./ g.d .^ 2))
    hx = sum(1 ./ Base.front(g.d) .^ 2)
    c = 0.0
    for k in H+1:H+g.n[N]
        h = min(g.zc[k+1] - g.zc[k], g.zc[k] - g.zc[k-1])
        c = max(c, med.vlayer[k] * dt * sqrt(hx + 1 / h^2))
    end
    return c
end

"Padded array index of the cell that contains coordinate `c` along axis `a` (z given as depth)."
cell_index(g::Grid{N}, a, c) where {N} =
    a < N || g.uniform ? ceil(Int, (c - g.beg[a]) / g.d[a]) + H : clamp(searchsortedfirst(g.zf, c), 1, length(g.zf))

"Thickness of cell `k` (padded index) in depth."
thickness(g::Grid, k) = g.uniform ? g.d[end] : g.zf[k] - g.zf[k-1]

"""
Coefficients of the depth derivatives (4th order: 9/8 and 1/24, and 2nd order) divided by the local spacing, per layer `k`:
`centre` for the derivatives of fields at the cell centres in depth (located at the faces k + 1/2) and `face` for those of
fields at the faces (located at the centres k). Constant on a uniform grid.
"""
function depth_coefficients(g::Grid{N}, ::Type{T}) where {N,T}
    m = g.n[N] + 2H
    if g.uniform                                    # the same constants as along x: identical results
        c = (a=fill(T(9 / 8 / g.d[N]), m), b=fill(T(1 / 24 / g.d[N]), m), c=fill(T(1 / g.d[N]), m))
        return c, c
    end
    zc, zf = g.zc, g.zf
    j4(x, k) = 2 <= k <= m - 2 ? 9 / 8 * (x[k+1] - x[k]) - 1 / 24 * (x[k+2] - x[k-1]) : x[min(k + 1, m)] - x[k]
    hf4 = [j4(zc, k) for k in 1:m]                                  # at the faces k + 1/2
    hf2 = [zc[min(k + 1, m)] - zc[k] for k in 1:m]
    hc2 = [k == 1 ? 2(zf[1] - zc[1]) : zf[k] - zf[k-1] for k in 1:m]  # at the centres k
    hc4 = [k == 1 ? hc2[1] : j4(zf, k - 1) for k in 1:m]
    hf2[m] = hf2[m-1]
    coef(h4, h2) = (a=T.(9 / 8 ./ h4), b=T.(1 / 24 ./ h4), c=T.(1 ./ h2))
    return coef(hf4, hf2), coef(hc4, hc2)
end

"Depth coefficients of layer `k` of the point `I` (a tuple `(9/8, 1/24, 2nd order) / spacing`); constant tuples along x, y."
@inline coefat(c::NTuple{3,Number}, I) = c
Base.@propagate_inbounds coefat(c::NamedTuple, I) = (k = last(Tuple(I)); (c.a[k], c.b[k], c.c[k]))

"""
The depth coefficients of the rows `k-2:k+2` (CPU loops over a row: loaded once per row instead of per cell; the compiler
cannot hoist them out of the loop, as they might alias the fields).
"""
struct RowCoef{T}
    a::NTuple{5,T}
    b::NTuple{5,T}
    c::NTuple{5,T}
    k::Int
end
Base.@propagate_inbounds coefat(r::RowCoef, I) = (d = last(Tuple(I)) - r.k + 3; (r.a[d], r.b[d], r.c[d]))
"""
The arguments of a kernel for the row `k`: depth coefficient arrays (NamedTuples `(a, b, c)`, also inside the coefficient
tuple `c`) as `RowCoef`. Two explicit levels, not recursion (Julia 1.10 gives up inferring recursion over nested tuples).
"""
Base.@propagate_inbounds atrow(args::Tuple, k) = map(x -> x isa Tuple ? map(y -> rowcoef(y, k), x) : rowcoef(x, k), args)
Base.@propagate_inbounds rowcoef(x::NamedTuple{(:a, :b, :c)}, k) =
    RowCoef(ntuple(d -> x.a[k-3+d], Val(5)), ntuple(d -> x.b[k-3+d], Val(5)), ntuple(d -> x.c[k-3+d], Val(5)), k)
@inline rowcoef(x, k) = x

"Field `A` as an internal array: z flipped so that k = 1 is the top, squeezed to (x, z) in 2D."
function internal(A::AbstractArray{Float64,3}, ::Val{N}) where {N}
    f = reverse(A, dims=3)
    return N == 3 ? f : f[:, 1, :]
end

"""
Pad `A` (physical cells `n`) by `H` ghost cells on every side by replication; the absorber zone (`na` cells at the sides
and the bottom) is homogenized, as in OpenSWPC.
"""
function pad_absorber(A::Array{Float64,N}, n::NTuple{N,Int}, na) where {N}
    P = Array{Float64,N}(undef, n .+ 2H)
    for I in CartesianIndices(P)
        J = ntuple(a -> a == N ? clamp(I[a] - H, 1, n[a] - na) : clamp(I[a] - H, na + 1, n[a] - na), Val(N))
        P[I] = A[J...]
    end
    return P
end

"""
Build the material arrays `(rho, lam, mu, taup, taus)` of the padded cells `W` (global index ranges: the whole grid, or the
cells of an MPI rank with a few around them), the 2nd-order zones near free surfaces `(fs_top, fs_bot, ob_top, ob_bot)` and
the maximum P velocity (`vlayer` per layer). `block(S)` gives the fields of the physical cells `S` (ranges in x, (y)) as
internal arrays. With `Val(true)` the arrays are returned as [`Window`](@ref)s, otherwise (the whole grid) as plain arrays.
"""
function setup_medium(block, g::Grid{N}, W::NTuple{N,UnitRange{Int}}, ::Val{windows}, na, fq_min, fq_max, fq_ref;
                      periodic=ntuple(_ -> false, Val(N - 1))) where {N,windows}
    n = g.n
    # the physical cell whose medium a padded cell gets: itself, or for the padding and the absorber zone (`na` cells at the
    # sides and the bottom, homogenized as in OpenSWPC) the nearest cell outside them; periodic axes: the cell on the other side
    J = ntuple(a -> [a == N ? clamp(i - H, 1, n[a] - na) : periodic[a] ? mod1(i - H, n[a]) : clamp(i - H, na + 1, n[a] - na) for i in W[a]], Val(N))
    S = ntuple(a -> minimum(J[a]):maximum(J[a]), Val(N - 1))
    b = block(S)
    pad(A) = A[ntuple(a -> a == N ? J[a] : J[a] .- (first(S[a]) - 1), Val(N))...]
    rho = pad(max.(b.rho, 0.001))   # air needs a small non-zero density (as in OpenSWPC)
    vp, vs, qp, qs = pad(b.vp), pad(b.vs), pad(b.Qp), pad(b.Qs)

    ts = relaxation_times(fq_min, fq_max)
    zeta = constq_zeta(fq_min, fq_max, ts)
    taup, taus = NM * zeta ./ qp, NM * zeta ./ qs
    Wc = Base.front(W)
    cells(A) = windows ? Window(A, W) : A
    cols(A, r=Wc) = windows ? Window(A, r) : A
    v = haskey(b, :epsilon) ? vti_medium(rho, vp, vs, pad(b.epsilon), pad(b.delta), pad(b.gamma), taup, taus, ts, fq_ref) : nothing   # VTI
    mu = v === nothing ? @.(rho * vs^2 / visco_chi((ts,), taus, fq_ref)^2) : v.mu       # unrelaxed moduli
    lam = v === nothing ? @.(rho * vp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu) : v.lam
    vfast = v === nothing ? @.(sqrt((lam + 2mu) / rho)) : v.vpmax                    # VTI: the fastest direction
    columns = CartesianIndices(Base.front(size(rho)))
    vlayer = [maximum(C -> vfast[C, k], columns) for k in axes(rho, N)]
    vti = v === nothing ? (;) : (; wx=cells(v.wx), wz=cells(v.wz), s13=cells(v.s13), mu66=cells(v.mu66))
    attenuated = any(!iszero, taup) || any(!iszero, taus)
    z = surface_zones(lam, mu, n, W)

    return (; rho=cells(rho), lam=cells(lam), mu=cells(mu), taup=cells(taup), taus=cells(taus), ts, vmax=maximum(vlayer), vlayer,
              attenuated, fs_top=cols(z.fs_top), fs_bot=cols(z.fs_bot), ob_top=cols(z.ob_top), ob_bot=cols(z.ob_bot),
              kfs=cols(z.kfs, z.held), kob=cols(z.kob, z.held), vti...)   # kfs, kob are indexed by padded column
end

"""
Free-surface (`fs`) and ocean-bottom (`ob`) zones, per column, where 2nd-order instead of
4th-order differences are used (OpenSWPC `surface_detection`). `lam`, `mu` are those of the padded cells `W` (global index
ranges); the zones are those of the padded columns `W` whose neighbours (2 before, 3 after) are among them, the others are empty.
"""
function surface_zones(lam, mu, n::NTuple{N}, W::NTuple{N,UnitRange{Int}}) where {N}
    ϵ = eps(Float32)
    nz = n[N]
    hn = Base.front(n)
    Wc = Base.front(W)                 # padded columns (physical column + H)
    o = first.(Wc) .- 1
    # kfs/kob: last k (physical) with vacuum above material; 0 if none. Detected on the physical columns 0:n+2, as in OpenSWPC;
    # indexed by padded column, up to the column n + 3 (one beyond the padding, never detected)
    held = map((r, m) -> first(r):(last(r) == m + 2H ? last(r) + 1 : last(r)), Wc, hn)
    kfs, kob = zeros(Int, length.(held)), zeros(Int, length.(held))
    for P in CartesianIndices(Wc)
        all(0 .<= Tuple(P) .- H .<= hn .+ 2) || continue
        col = CartesianIndex(Tuple(P) .- o)
        for k in 1:nz-1
            abs(mu[col, k+H]) < ϵ && abs(mu[col, k+1+H]) > ϵ && (kob[col] = k)
            abs(lam[col, k+H]) < ϵ && abs(lam[col, k+1+H]) > ϵ && (kfs[col] = k)
        end
    end
    # zone = padded k-range [top, bot]; ghost columns get an empty zone (top > bot)
    function zone(kk)
        top, bot = fill(Int32(1), length.(Wc)), fill(Int32(0), length.(Wc))
        for P in CartesianIndices(Wc)
            p = Tuple(P)
            all(1 .<= p .- H .<= hn) || continue
            all(map((q, r) -> first(r) <= q - 2 && q + 3 <= last(r), p, held)) || continue   # a column of another rank
            w = kk[CartesianIndices(map(q -> q-2:q+3, p .- o))]   # physical columns i-2 : i+3
            col = CartesianIndex(p .- o)
            top[col] = max(minimum(w) - 2, 1) + H
            bot[col] = min(maximum(w) + 2, nz) + H
        end
        return top, bot
    end
    (fs_top, fs_bot), (ob_top, ob_bot) = zone(kfs), zone(kob)
    return (; fs_top, fs_bot, ob_top, ob_bot, kfs, kob, held)
end

"""
Cerjan sponge factors `(centre, boundary)` along one axis of `n` cells (padded). No damping at the top (`top=false`).
"""
function cerjan(n, d, na; top=true)
    α, L = 0.09, na * d
    gc, gb = ones(n + 2H), ones(n + 2H)
    for i in 1:n
        if i <= na && top
            gc[i+H] = exp(-α * (1 - (i - 0.5) * d / L)^2)
            gb[i+H] = exp(-α * (1 - i * d / L)^2)
        elseif i >= n - na + 1
            gc[i+H] = exp(-α * (1 - (n - i + 1) * d / L)^2)
            gb[i+H] = exp(-α * (1 - (n - i + 0.5) * d / L)^2)
        end
    end
    return gc, gb
end
