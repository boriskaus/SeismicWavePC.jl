# 2D P-SV on a terrain-following (curvilinear) grid: summation-by-parts (SBP) finite differences with simultaneous
# approximation terms (SAT). A deformed 2D CartData (vertical columns, a grid line along the surface and along every
# major interface) is split into blocks at the interfaces (found from the material). In every block: collocated
# velocity-stress, dual-pairing upwind SBP operators (SummationByPartsOperators.jl, see sbp_coefficients.jl): the
# divergence (momentum) with D-, the gradient
# (stress) with D+, which are adjoint in the SBP norm, so that the semi-discrete energy changes only through the boundary
# terms; SAT terms make the top a free surface, couple the blocks (continuity of velocity and traction, energy-neutral)
# and absorb at the sides and the bottom (impedance, σ·n = -Z v). Leapfrog in time (velocities at half steps), the time
# step from the spectral radius of the operator. Attenuation as in the staggered solver (generalized Zener body).

include("sbp_coefficients.jl")

# ------------------------------------------------------------------ SBP operators on a 1D grid of n points

"Nodes of the SBP grid of `n` points on [a, b] and its interior spacing h."
function sbp_grid(a, b, n)
    n >= SBP_NMIN || error("an SBP grid needs at least $SBP_NMIN points")
    h = (b - a) / (2SBP_P[end] + n - 2SBP_NP + 1)
    x = [a + h * (j <= SBP_NP ? SBP_P[j] : SBP_P[end] + (j - SBP_NP)) for j in 1:n]
    for j in 1:SBP_NP
        x[n+1-j] = b - h * SBP_P[j]
    end
    return x, h
end
"Weights of the SBP norm (quadrature) on a grid of `n` points with interior spacing `h`."
sbp_weights(n, h) = [h * (j <= SBP_NP ? SBP_W[j] : j > n - SBP_NP ? SBP_W[n+1-j] : 1.0) for j in 1:n]

@assert length(SBP_OM) == 8 && length(SBP_OP) == 8                  # (the unrolled interior sums below)
"h × the boundary row `i` (1…NB) of D₋ (`tab = :LM`) or D₊ (`:LP`) applied to `g`: an if-chain over the rows with the
coefficients as literals (a runtime index into the tables would put them on the stack at every call; also for GPUs)."
@generated function boundary_row(::Val{tab}, g::G, i, ::Type{T}) where {tab,G,T}
    rows = tab === :LM ? SBP_LM : SBP_LP
    ex = :(zero(T))
    for r in length(rows):-1:1
        terms = [:($(T(rows[r][j])) * g($j)) for j in eachindex(rows[r]) if rows[r][j] != 0]
        ex = :(i == $r ? $(Expr(:call, :+, terms...)) : $ex)
    end
    return ex
end
"h (D₋ g)(i) on `n` points; `g(j)` is the value at node j."
@inline function dminus(g::G, i, n) where {G}
    T = typeof(g(i))
    i <= SBP_NB && return boundary_row(Val(:LM), g, i, T)
    i > n - SBP_NB && return -boundary_row(Val(:LP), j -> g(n + 1 - j), n + 1 - i, T)
    return dminus_int(g, i, T)
end
"h (D₊ g)(i) on `n` points."
@inline function dplus(g::G, i, n) where {G}
    T = typeof(g(i))
    i <= SBP_NB && return boundary_row(Val(:LP), g, i, T)
    i > n - SBP_NB && return -boundary_row(Val(:LM), j -> g(n + 1 - j), n + 1 - i, T)
    return dplus_int(g, i, T)
end

"h (D₋ g)(i) and h (D₊ g)(i) away from the boundaries (no branches: the interior of a block vectorizes)."
@inline function dminus_int(g::G, i, ::Type{T}) where {G,T}
    s = zero(T)
    Base.Cartesian.@nexprs 8 m -> (s += T(SBP_CM[m]) * g(i + SBP_OM[m]))
    return s
end
@inline function dplus_int(g::G, i, ::Type{T}) where {G,T}
    s = zero(T)
    Base.Cartesian.@nexprs 8 m -> (s += T(SBP_CP[m]) * g(i + SBP_OP[m]))
    return s
end

# ------------------------------------------------------------------ the mesh

"""
    mesh = curvilinear_mesh(x, surfaces; dz, bottom)

A deformed 2D `CartData` (size nx × 1 × nz, km) for `simulate` on a terrain-following grid: one grid line follows each
of the `surfaces`, top first: the topography, then interfaces with a jump in the material (e.g. a Moho), each a function
z(x) [km] or a surface `CartData` (nx × 1 × 1, linearly interpolated). The bottom is flat at `bottom` [km]; between two
surfaces z is blended linearly. The columns are vertical, at the nodes of the SBP operator over `first(x)`…`last(x)` with
`length(x)` points (equidistant except the outermost few); between two surfaces at least 13 nodes, with about `dz` [km]
spacing where the layer is thickest.

Fill the fields `rho` [kg/m³], `vp`, `vs` [km/s] (and optionally `Qp`, `Qs`) at the nodes with the usual GMG tools,
e.g. layers with `Z .< moho.(X)` and a magma chamber with `add_ellipsoid!`, then `simulate(addfield(mesh, (; rho, vp, vs)); ...)`.
"""
function curvilinear_mesh(x, surfaces; dz, bottom)
    nx = length(x)
    gx, _ = sbp_grid(Float64(first(x)), Float64(last(x)), nx)
    surf(f::Function) = f.(gx)
    function surf(c::CartData)                                          # linear interpolation of a surface CartData
        xs, zs = c.x.val[:, 1, 1], c.z.val[:, 1, 1]
        p = sortperm(xs)
        xs, zs = xs[p], zs[p]
        lin(v) = (j = clamp(searchsortedlast(xs, v), 1, length(xs) - 1); zs[j] + (zs[j+1] - zs[j]) * (v - xs[j]) / (xs[j+1] - xs[j]))
        return lin.(gx)
    end
    S = [[surf(f) for f in surfaces]; [fill(Float64(bottom), nx)]]       # top to bottom
    Zs = Matrix{Float64}[]
    for b in length(S)-1:-1:1                                          # bottom first (z increases with the index)
        top, bot = S[b], S[b+1]
        all(top .> bot) || error("curvilinear_mesh: surfaces $b and $(b+1) touch or cross")
        nz = max(SBP_NMIN, ceil(Int, maximum(top .- bot) / dz) + 1)
        η, _ = sbp_grid(0.0, 1.0, nz)
        Zb = [bot[i] + (top[i] - bot[i]) * e for i in 1:nx, e in η]
        Zb[:, end] .= top                                              # exactly on the surface
        push!(Zs, isempty(Zs) ? Zb : Zb[:, 2:end])                     # the shared row once
    end
    Z2 = reduce(hcat, Zs)
    nz = size(Z2, 2)
    X = [gx[i] for i in 1:nx, _ in 1:1, _ in 1:nz]
    Z = reshape(Z2, nx, 1, nz)
    return CartData(X, zero(X), Z, (; z=copy(Z)))
end

"""
    mesh = curvilinear_mesh(x, y, surfaces; dz, bottom)

The 3D version: columns at the SBP nodes in x and y, `surfaces` functions z(x, y) [km] or surface `CartData`
(nx × ny × 1, bilinearly interpolated).
"""
function curvilinear_mesh(x, y, surfaces; dz, bottom)
    nx, ny = length(x), length(y)
    gx, _ = sbp_grid(Float64(first(x)), Float64(last(x)), nx)
    gy, _ = sbp_grid(Float64(first(y)), Float64(last(y)), ny)
    surf(f::Function) = [f(a, b) for a in gx, b in gy]
    function surf(c::CartData)                                          # bilinear interpolation of a surface CartData
        xs, ys, zs = c.x.val[:, 1, 1], c.y.val[1, :, 1], c.z.val[:, :, 1]
        ix(v, g) = (q = clamp(searchsortedlast(g, v), 1, length(g) - 1); (q, (v - g[q]) / (g[q+1] - g[q])))
        return [(((p, s), (q, t)) = (ix(a, xs), ix(b, ys));
                 (1 - s) * (1 - t) * zs[p, q] + s * (1 - t) * zs[p+1, q] + (1 - s) * t * zs[p, q+1] + s * t * zs[p+1, q+1]) for a in gx, b in gy]
    end
    S = [[surf(f) for f in surfaces]; [fill(Float64(bottom), nx, ny)]]
    Zs = Array{Float64,3}[]
    for b in length(S)-1:-1:1
        top, bot = S[b], S[b+1]
        all(top .> bot) || error("curvilinear_mesh: surfaces $b and $(b+1) touch or cross")
        nz = max(SBP_NMIN, ceil(Int, maximum(top .- bot) / dz) + 1)
        η, _ = sbp_grid(0.0, 1.0, nz)
        Zb = [bot[i, j] + (top[i, j] - bot[i, j]) * e for i in 1:nx, j in 1:ny, e in η]
        Zb[:, :, end] .= top
        push!(Zs, isempty(Zs) ? Zb : Zb[:, :, 2:end])
    end
    Z = cat(Zs...; dims=3)
    nz = size(Z, 3)
    X = [gx[i] for i in 1:nx, _ in 1:ny, _ in 1:nz]
    Y = [gy[j] for _ in 1:nx, j in 1:ny, _ in 1:nz]
    return CartData(X, Y, Z, (; z=copy(Z)))
end

"Is the `CartData` a deformed (terrain-following) mesh, i.e. do its rows of nodes vary in depth?"
function is_curvilinear(model::CartData)
    Z = model.z.val
    return any(k -> maximum(view(Z, :, :, k)) - minimum(view(Z, :, :, k)) > 1e-9 * (maximum(Z) - minimum(Z)), axes(Z, 3))
end

"""
    rows, below = find_blocks(Z, vp, vs; jump=0.05)

Interfaces of a deformed mesh (nx × nz, z increasing with the index) from its material: a row r is an interface if vp or
vs jumps along the full width, in every column between r - 1 and r or between r and r + 1; a jump is a relative change of
more than `jump` and more than 3 times the changes between the neighbouring rows. Jumps over part of the width (a magma
chamber) and gradients (also steep ones) are no interfaces. If all columns jump between the
same two rows, the mesh decides (the shape of the vertical spacing changes at the grid line that follows the interface);
if it does not, the interface is the upper row (GMG: a node exactly on a surface is not below it). Returns the rows, and
for each whether each column's node has the material below the interface (true) or above it.
"""
function find_blocks(Z, vp, vs; jump=0.05, own=axes(vp, 1), red=nothing)
    rel(a, b) = a == b ? 0.0 : abs(a - b) / max(abs(a), abs(b))      # (0 between fluid nodes, vs = 0)
    nx, nz = size(vp)
    # a jump: larger than `jump` and than 3 times the changes between the neighbouring rows (not a steep gradient)
    ch(A, i, k) = 1 <= k <= nz - 1 ? rel(A[i, k], A[i, k+1]) : 0.0
    isjump(A, i, k) = ch(A, i, k) > jump && ch(A, i, k) > 3max(ch(A, i, k - 1), ch(A, i, k + 1))
    J = [isjump(vp, i, k) || isjump(vs, i, k) for i in 1:nx, k in 1:nz-1]
    # (with MPI: the decisions over the columns `own` of all processes, `red` reduces over them: and, sum, max, count)
    and(v) = red === nothing ? v : red.and(v)
    sm(x) = red === nothing ? x : red.sum(x)
    mx(x) = red === nothing ? x : red.max(x)
    ncols = red === nothing ? length(own) : red.sum(length(own))
    allrows = and([all(i -> J[i, r-1] || J[i, r], own) for r in 2:nz-1])
    cand = [r for (q, r) in enumerate(2:nz-1) if allrows[q]]
    shape(k) = (d = Z[own, k+1] .- Z[own, k]; d ./ sm(sum(d)))
    kink(k) = 2 <= k <= nz - 1 ? mx(maximum(abs, shape(k) .- shape(k - 1); init=0.0)) * ncols : 0.0
    rows = Int[]
    for r in cand
        (r in rows || r - 1 in rows) && continue
        both = r + 1 in cand && and([all(i -> J[i, r], own)])[1]          # (the jump is between rows r and r + 1 everywhere)
        # (kinks at the level of rounding, e.g. of a flat mesh, are none: then the upper row, as GMG)
        push!(rows, !both ? r : kink(r) > 1e-9 && kink(r) > 3kink(r + 1) ? r : r + 1)
    end
    return rows, [[J[i, r] for i in 1:nx] for r in rows]
end
"""
Is a block fluid (vs = 0)? Blocks are fluid or solid as a whole: fluid–solid interfaces must be interfaces of the mesh (`vs` of
the columns of this process; with MPI: of all).
"""
function block_fluid(vs, b, par)
    r = (Int(any(iszero, vs)), Int(any(>(0), vs)))
    f, s = par === nothing ? r : (MPI.Allreduce(r[1], max, par.comm), MPI.Allreduce(r[2], max, par.comm))
    f == 1 && s == 1 && error("curvilinear grids: block $b is part fluid (vs = 0), part solid: fluid regions must be whole layers " *
                              "of the mesh, between two of its surfaces (fluid–solid interfaces: interfaces of the mesh)")
    return f == 1
end
"The reductions over the processes of `find_blocks` (nothing: one process)."
block_reducer(par) = par === nothing ? nothing : (; and=v -> MPI.Allreduce(Int.(v), min, par.comm) .== 1,
                                                    sum=x -> MPI.Allreduce(x, +, par.comm), max=x -> MPI.Allreduce(x, max, par.comm))

"""
The columns `ix` (, `iy`) of a curvilinear model: heights [km] and the fields, as arrays (columns..., nz) (2D: (nx, nz)).
`CartData`: cut out; [`DistributedModel`](@ref): built by its function.
"""
function model_columns(model::CartData, ix, iy, N)
    cut(A) = N == 2 ? A[ix, 1, :] : A[ix, iy, :]
    return cut(model.z.val), map(cut, model.fields)
end
function model_columns(model::DistributedModel, ix, iy, N)
    b = model.f(ix, N == 2 ? (1:1) : iy)
    fields = b isa CartData ? b.fields : b
    Z = b isa CartData ? b.z.val : haskey(fields, :Z) ? fields.Z : error("DistributedModel: give the heights `Z` of the columns")
    sz = (length(ix), N == 2 ? 1 : length(iy), model.nzc)
    size(Z) == sz || error("DistributedModel: the columns ($ix, $iy) have size $(size(Z)), expected $sz")
    cut(A) = (size(A) == sz || error("DistributedModel: a field of the columns ($ix, $iy) has size $(size(A)), expected $sz");
              N == 2 ? Float64.(A[:, 1, :]) : Float64.(A))
    return cut(Z), map(cut, fields)
end

# ------------------------------------------------------------------ kernels (one node (i, k) of a block)

"""
The arrays `names` of every block (those present): kernels that see all blocks get only what they read, as their arguments
are limited (Metal: 4 kB; with attenuation and VTI a 3D block has ~40 arrays).
"""
slim(blocks, names) = map(B -> NamedTuple{Tuple(k for k in names if haskey(B, k))}(B), blocks)
const CV_VEL = (:V1, :V2, :V3)
const CV_STRESS = (:S11, :S22, :S33, :S23, :S13, :S12, :S1, :S2)
const CV_RECORD = (CV_VEL..., CV_STRESS..., :lam, :mu, :mu66)                       # (stations)
const CV_FORCE = (CV_VEL..., :iρJ)                                                  # (forces)
const CV_METRIC = (:J, :Jx, :Jy, :hξ, :hψ, :hη, :ox, :oy)
const CV_TF = (CV_FORCE..., CV_METRIC...)                                           # (teleseismic momentum)
const CV_UPDATE = (CV_STRESS..., CV_METRIC..., :V1, :lam, :mu, :mu66, :tp, :ts, :wx, :wz, :s13, :tilt, :azim,
                   :R11, :R22, :R33, :R23, :R13, :R12, :R1, :R2)                    # (the stress update: teleseismic stress)

"The arrays of a neighbouring block that the interface terms read (small kernel arguments: Metal allows 4 kB)."
neighbour(B) = (; V1=B.V1, V2=B.V2, S11=B.S11, S12=B.S12, S22=B.S22, Jx=B.Jx, fluid=B.fluid)
neighbour(::Nothing) = nothing

# Fluid–solid interfaces (`B.fluid`: a block of fluid, μ = 0; an interface with fluid on one side): slip. Only the normal
# components are coupled (half the mismatch of the normal traction and velocity, as welded), the tangential tractions removed
# on both sides (as at a free surface; zero in the fluid anyway). The normal n ∝ m = Jη (the same on both sides: the metric
# along the row), n nᵀ = m mᵀ / |m|². The SBP identity of the adjoint holds as for welded interfaces and free surfaces.
"The traction mismatch `e` of the momentum SAT at an interface (`t` own, `o` of the other block; top row: -e/(2Hη), bottom: +e/(2Hη))."
@inline function sat_traction(slip, m, t, o)
    slip || return t .- o
    s = sum(m .* (t .+ o)) / sum(m .* m)
    return 2 .* t .- s .* m
end
"The velocity mismatch of the stress SAT at an interface (welded: all components, slip: the normal one)."
@inline function sat_velocity(slip, m, v, o)
    slip || return v .- o
    s = sum(m .* (v .- o)) / sum(m .* m)
    return s .* m
end
@inline orzero(x, I) = x[I]
@inline orzero(::Nothing, I) = false
@inline cmemvar!(R, I, f, vc, dt) = memvar!(R, I, f, vc, dt)
@inline cmemvar!(R, I, f, ::Nothing, dt) = zero(f)
@inline cmemvar!(::Nothing, I, f, ::Nothing, dt) = zero(f)

# Parts of the operators: every derivative is the sum of its ξ part (with the side SAT terms) and its η part (with the SAT
# terms of the free surface, the bottom and the interfaces). The adjoint applies them separately (`B.part` = Val(:xi) or
# Val(:eta); default Val(:all)): the SBP identity H A = -(H J G)ᵀ holds for each direction.
@inline part(B::NamedTuple{K}) where {K} = :part in K ? B.part : Val(:all)
@inline usexi(::Val{p}) where {p} = p === :all || p === :xi
@inline usepsi(::Val{p}) where {p} = p === :all || p === :psi
@inline useeta(::Val{p}) where {p} = p === :all || p === :eta
# PML (absorber = :pml): the ξ parts (in the side layers) and the η parts (in the bottom layer) are stretched with the ADE of
# the CFS-PML of OpenSWPC (`damping_profile`): γ₁ d + γ₂ m, m ← γ₃ m + γ₄ dt d. `B.pml`: the profiles γ (`gx` at the
# global nodes in x, `gz` at the rows of the block) and the memory variables (side layers: 2np columns, as `xslab`; bottom:
# the rows of the bottom block). Regions flag their nodes in the layers (`Val((xint, px, pz))`).
@inline pmlof(B::NamedTuple{K}) where {K} = :pml in K ? B.pml : nothing
@inline xslab(P, ig) = ig <= P.np ? ig : ig - P.nxg + 2P.np
@inline Base.@propagate_inbounds function pml_stretch(m, I, γ, d, dt)
    q = m[I]
    m[I] = γ[3] * q + γ[4] * dt * d
    return γ[1] * d + γ[2] * q
end
# M-PML (absorber = :mpml, as the staggered grid): every derivative in the layers is damped with d + p d∥, d∥ the damping of
# the other axes (p = `P.mpml`); α and κ of its own axis. The profiles are then (d, α, 1/κ, d) (`cfs_profile`) at all nodes,
# the coefficients computed per node, and the memory variables of the ξ parts also in the bottom layer away from the sides
# (`…b`, rows of the bottom block) and those of the η parts also in the side layers (`…s`, as `xslab`).
@inline ismpml(P::NamedTuple{K}) where {K} = :mpml in K
"""
The ADE coefficients of the M-PML from the profile `g` = (d, α, 1/κ, d) of the axis and the damping `dpar` of the others, for
the time step `dt` (the kernels: `P.mdt[1]`, the time step of the simulation, also when the adjoint applies them with dt = 1).
"""
@inline function mpml_gamma(P, g, dpar, dt)
    d, α, iκ, _ = g
    u = (d + P.mpml * dpar) * iκ
    c = dt / 2
    iq = inv(1 + c * (α + u))
    return ((1 + c * α) * iκ * iq, -iκ * iq, (1 - c * (α + u)) * iq, u * iq)
end
"The ADE coefficients of a ξ (`pml_xgamma`) or η (`pml_zgamma`) part at the node (global column `ig`, row `k`)."
@inline Base.@propagate_inbounds pml_xgamma(P, ig, k, dt) = ismpml(P) ? mpml_gamma(P, P.gx[ig], P.gz[k][4], P.mdt[1]) : P.gx[ig]
@inline Base.@propagate_inbounds pml_zgamma(P, ig, k, dt) = ismpml(P) ? mpml_gamma(P, P.gz[k], P.gx[ig][4], P.mdt[1]) : P.gz[k]
"The memory variable (array, index) of a ξ part (`a`: the side layers, `ab`: M-PML, the bottom layer between them)."
@inline Base.@propagate_inbounds function pml_xmem(P, ::Val{a}, ::Val{ab}, i, ig, k) where {a,ab}
    (!ismpml(P) || ig <= P.np || ig > P.nxg - P.np) && return getfield(P, a), CartesianIndex(xslab(P, ig), k)
    return getfield(P, ab), CartesianIndex(i, k)
end
"The memory variable (array, index) of an η part (`a`: the bottom layer, `as`: M-PML, the side layers above it)."
@inline Base.@propagate_inbounds function pml_zmem(P, ::Val{a}, ::Val{as}, i, ig, k) where {a,as}
    (!ismpml(P) || k <= size(getfield(P, a), 2)) && return getfield(P, a), CartesianIndex(i, k)
    return getfield(P, as), CartesianIndex(xslab(P, ig), k)
end
@inline flags(F) = F isa Tuple ? F : (F, false, false)
"The PML memory variables of a block (checkpoints)."
pml_arrays(B) = (P = pmlof(B); P === nothing ? () : Tuple(v for (k, v) in pairs(P) if v isa AbstractArray && !(k in (:gx, :gy, :gz, :mdt))))

# Momentum: ρ J ∂t v_i = Dξ(Jξ_j σ_ij) + Dη(Jη_j σ_ij), Jξ = (J, 0), Jη = (Jx, 1), with SAT terms on the boundary nodes:
# the traction removed (free surface on top; free or absorbing at the sides and bottom), half the traction mismatch at
# interfaces (`Bd`, `Bu`: the blocks below and above, `nothing` at the bottom and top). The impedance damping `a` is
# implicit (trapezoidal): unconditionally stable. `damp = false` leaves it out (power iteration).
# `Val(true)`: a node away from the sides (interior stencils in x, no side terms): the boundary rows then vectorize.
@inline Base.@propagate_inbounds function momentum_cv!(B, Bd, Bu, dt, damp, ::Val{F}, i, k) where {F}
    xint, px, pz = flags(F)
    (; V1, V2, S11, S22, S12, J, Jx, iρJ, a1, a2, hξ, hη, ox, nxg) = B
    nz = size(V1, 2)
    T = eltype(V1)
    w = T(SBP_W[1])
    # x: local index i, global i + ox of nxg (MPI subdomains); the boundary closures in global indices
    g1, g2 = j -> @inbounds(J[j, k] * S11[j, k]), j -> @inbounds(J[j, k] * S12[j, k])
    x1 = (xint ? dminus_int(g1, i, T) : dminus(q -> g1(q - ox), i + ox, nxg)) / hξ
    x2 = (xint ? dminus_int(g2, i, T) : dminus(q -> g2(q - ox), i + ox, nxg)) / hξ
    if xint
    elseif i + ox == 1
        x1 += J[i, k] * S11[i, k] / (w * hξ); x2 += J[i, k] * S12[i, k] / (w * hξ)
    elseif i + ox == nxg
        x1 -= J[i, k] * S11[i, k] / (w * hξ); x2 -= J[i, k] * S12[i, k] / (w * hξ)
    end
    z1 = dminus(j -> @inbounds(Jx[i, j] * S11[i, j] + S12[i, j]), k, nz) / hη
    z2 = dminus(j -> @inbounds(Jx[i, j] * S12[i, j] + S22[i, j]), k, nz) / hη
    if k == nz
        t1, t2 = Jx[i, k] * S11[i, k] + S12[i, k], Jx[i, k] * S12[i, k] + S22[i, k]
        if Bu === nothing
            z1 -= t1 / (w * hη); z2 -= t2 / (w * hη)
        else
            e1, e2 = sat_traction(B.fluid != Bu.fluid, (Jx[i, k], one(T)), (t1, t2),
                                  (Bu.Jx[i, 1] * Bu.S11[i, 1] + Bu.S12[i, 1], Bu.Jx[i, 1] * Bu.S12[i, 1] + Bu.S22[i, 1]))
            z1 -= e1 / (2w * hη); z2 -= e2 / (2w * hη)
        end
    end
    if k == 1
        t1, t2 = Jx[i, k] * S11[i, k] + S12[i, k], Jx[i, k] * S12[i, k] + S22[i, k]
        if Bd === nothing
            z1 += t1 / (w * hη); z2 += t2 / (w * hη)
        else
            n = size(Bd.V1, 2)
            e1, e2 = sat_traction(B.fluid != Bd.fluid, (Jx[i, k], one(T)), (t1, t2),
                                  (Bd.Jx[i, n] * Bd.S11[i, n] + Bd.S12[i, n], Bd.Jx[i, n] * Bd.S12[i, n] + Bd.S22[i, n]))
            z1 += e1 / (2w * hη); z2 += e2 / (2w * hη)
        end
    end
    p = part(B)
    usexi(p) || (x1 = x2 = zero(T))
    useeta(p) || (z1 = z2 = zero(T))
    P = pmlof(B)
    if px && P !== nothing
        γ = pml_xgamma(P, i + ox, k, dt)
        x1 = pml_stretch(pml_xmem(P, Val(:xv1), Val(:xv1b), i, i + ox, k)..., γ, x1, dt)
        x2 = pml_stretch(pml_xmem(P, Val(:xv2), Val(:xv2b), i, i + ox, k)..., γ, x2, dt)
    end
    if pz && P !== nothing
        γ = pml_zgamma(P, i + ox, k, dt)
        z1 = pml_stretch(pml_zmem(P, Val(:zv1), Val(:zv1s), i, i + ox, k)..., γ, z1, dt)
        z2 = pml_stretch(pml_zmem(P, Val(:zv2), Val(:zv2s), i, i + ox, k)..., γ, z2, dt)
    end
    d1, d2 = x1 + z1, x2 + z2
    b1, b2 = damp ? (a1[i, k] * dt / 2, a2[i, k] * dt / 2) : (zero(T), zero(T))
    V1[i, k] = ((1 - b1) * V1[i, k] + dt * iρJ[i, k] * d1) / (1 + b1)
    V2[i, k] = ((1 - b2) * V2[i, k] + dt * iρJ[i, k] * d2) / (1 + b2)
end

# The interior of a block (no boundary closures, SAT terms, damping or PML): the same updates without branches.
@inline Base.@propagate_inbounds function momentum_cv_int!(B, dt, i, k)
    (; V1, V2, S11, S22, S12, J, Jx, iρJ, hξ, hη) = B
    T = eltype(V1)
    p = part(B)
    d1 = (usexi(p) ? dminus_int(j -> @inbounds(J[j, k] * S11[j, k]), i, T) / hξ : zero(T)) +
         (useeta(p) ? dminus_int(j -> @inbounds(Jx[i, j] * S11[i, j] + S12[i, j]), k, T) / hη : zero(T))
    d2 = (usexi(p) ? dminus_int(j -> @inbounds(J[j, k] * S12[j, k]), i, T) / hξ : zero(T)) +
         (useeta(p) ? dminus_int(j -> @inbounds(Jx[i, j] * S12[i, j] + S22[i, j]), k, T) / hη : zero(T))
    V1[i, k] += dt * iρJ[i, k] * d1
    V2[i, k] += dt * iρJ[i, k] * d2
end
@inline Base.@propagate_inbounds function stress_cv_int!(B, vc, dt, i, k)
    (; V1, V2, S11, S22, S12, J, Jx, lam, mu, hξ, hη) = B
    T = eltype(V1)
    p = part(B)
    a = usexi(p) ? dplus_int(j -> @inbounds(V1[j, k]), i, T) / hξ : zero(T)
    b = useeta(p) ? dplus_int(j -> @inbounds(V1[i, j]), k, T) / hη : zero(T)
    c = usexi(p) ? dplus_int(j -> @inbounds(V2[j, k]), i, T) / hξ : zero(T)
    e = useeta(p) ? dplus_int(j -> @inbounds(V2[i, j]), k, T) / hη : zero(T)
    stress_update_cv!(B, vc, dt, i, k, a, b, c, e)
end

# the interior as a region of the same kernels (one launch for all regions of a block, see foreach_region!)
@inline Base.@propagate_inbounds momentum_cv!(B, Bd, Bu, dt, damp, ::Val{:int}, i, k) = momentum_cv_int!(B, dt, i, k)
@inline Base.@propagate_inbounds stress_cv!(B, Bd, Bu, vc, dt, ::Val{:int}, i, k) = stress_cv_int!(B, vc, dt, i, k)

# Stress: the gradient ∂x = Dξ + (Jx / J) Dη, ∂z = Dη / J with D₊ (half the velocity mismatch at interfaces added to Dη v),
# and the stress update of the staggered solver (memory variables of the generalized Zener body if `vc !== nothing`).
@inline Base.@propagate_inbounds function stress_cv!(B, Bd, Bu, vc, dt, ::Val{F}, i, k) where {F}
    xint, px, pz = flags(F)
    (; V1, V2, hξ, hη) = B
    nx, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    g1, g2 = j -> @inbounds(V1[j, k]), j -> @inbounds(V2[j, k])
    ox, nxg = B.ox, B.nxg
    a = (xint ? dplus_int(g1, i, T) : dplus(q -> g1(q - ox), i + ox, nxg)) / hξ
    b = dplus(j -> @inbounds(V1[i, j]), k, nz) / hη
    c = (xint ? dplus_int(g2, i, T) : dplus(q -> g2(q - ox), i + ox, nxg)) / hξ
    e = dplus(j -> @inbounds(V2[i, j]), k, nz) / hη
    if k == nz && Bu !== nothing
        u1, u2 = sat_velocity(B.fluid != Bu.fluid, (B.Jx[i, k], one(T)), (V1[i, k], V2[i, k]), (Bu.V1[i, 1], Bu.V2[i, 1]))
        b -= u1 / (2w * hη); e -= u2 / (2w * hη)
    end
    if k == 1 && Bd !== nothing
        n = size(Bd.V1, 2)
        u1, u2 = sat_velocity(B.fluid != Bd.fluid, (B.Jx[i, k], one(T)), (V1[i, k], V2[i, k]), (Bd.V1[i, n], Bd.V2[i, n]))
        b += u1 / (2w * hη); e += u2 / (2w * hη)
    end
    p = part(B)
    usexi(p) || (a = c = zero(T))
    useeta(p) || (b = e = zero(T))
    P = pmlof(B)
    if px && P !== nothing
        γ = pml_xgamma(P, i + ox, k, dt)
        a = pml_stretch(pml_xmem(P, Val(:xa), Val(:xab), i, i + ox, k)..., γ, a, dt)
        c = pml_stretch(pml_xmem(P, Val(:xc), Val(:xcb), i, i + ox, k)..., γ, c, dt)
    end
    if pz && P !== nothing
        γ = pml_zgamma(P, i + ox, k, dt)
        b = pml_stretch(pml_zmem(P, Val(:zb), Val(:zbs), i, i + ox, k)..., γ, b, dt)
        e = pml_stretch(pml_zmem(P, Val(:ze), Val(:zes), i, i + ox, k)..., γ, e, dt)
    end
    stress_update_cv!(B, vc, dt, i, k, a, b, c, e)
end
# ------------------------------------------------------------------ SH (2D): vy (V1), σxy (S1), σyz (S2)
# Momentum: ρ J ∂t vy = Dξ(J σxy) + Dη(Jx σxy + σyz), the SAT terms as in P-SV; stress: σxy with C66 (`mu66`), σyz with
# C44 (`mu`), memory variables R1, R2. The parts and the PML as in P-SV (memory: xv1, xa (Dξ vy), zv1, zb (Dη vy)).
neighbour_sh(B) = (; V1=B.V1, S1=B.S1, S2=B.S2, Jx=B.Jx, fluid=B.fluid)
neighbour_sh(::Nothing) = nothing
@inline Base.@propagate_inbounds function momentum_sh!(B, Bd, Bu, dt, damp, ::Val{F}, i, k) where {F}
    xint, px, pz = flags(F)
    (; V1, S1, S2, J, Jx, iρJ, a1, hξ, hη, ox, nxg) = B
    nz = size(V1, 2)
    T = eltype(V1)
    w = T(SBP_W[1])
    g = j -> @inbounds(J[j, k] * S1[j, k])
    x1 = (xint ? dminus_int(g, i, T) : dminus(q -> g(q - ox), i + ox, nxg)) / hξ
    if xint
    elseif i + ox == 1
        x1 += J[i, k] * S1[i, k] / (w * hξ)
    elseif i + ox == nxg
        x1 -= J[i, k] * S1[i, k] / (w * hξ)
    end
    z1 = dminus(j -> @inbounds(Jx[i, j] * S1[i, j] + S2[i, j]), k, nz) / hη
    if k == nz
        t1 = Jx[i, k] * S1[i, k] + S2[i, k]
        # (fluid–solid: vy is tangential, the interface free on both sides)
        z1 -= Bu === nothing || B.fluid != Bu.fluid ? t1 / (w * hη) : (t1 - (Bu.Jx[i, 1] * Bu.S1[i, 1] + Bu.S2[i, 1])) / (2w * hη)
    end
    if k == 1
        t1 = Jx[i, k] * S1[i, k] + S2[i, k]
        if Bd === nothing || B.fluid != Bd.fluid
            z1 += t1 / (w * hη)
        else
            n = size(Bd.V1, 2)
            z1 -= ((Bd.Jx[i, n] * Bd.S1[i, n] + Bd.S2[i, n]) - t1) / (2w * hη)
        end
    end
    p = part(B)
    usexi(p) || (x1 = zero(T))
    useeta(p) || (z1 = zero(T))
    P = pmlof(B)
    px && P !== nothing && (x1 = pml_stretch(pml_xmem(P, Val(:xv1), Val(:xv1b), i, i + ox, k)..., pml_xgamma(P, i + ox, k, dt), x1, dt))
    pz && P !== nothing && (z1 = pml_stretch(pml_zmem(P, Val(:zv1), Val(:zv1s), i, i + ox, k)..., pml_zgamma(P, i + ox, k, dt), z1, dt))
    b1 = damp ? a1[i, k] * dt / 2 : zero(T)
    V1[i, k] = ((1 - b1) * V1[i, k] + dt * iρJ[i, k] * (x1 + z1)) / (1 + b1)
end
@inline Base.@propagate_inbounds function momentum_sh!(B, Bd, Bu, dt, damp, ::Val{:int}, i, k)
    (; V1, S1, S2, J, Jx, iρJ, hξ, hη) = B
    T = eltype(V1)
    p = part(B)
    d = (usexi(p) ? dminus_int(j -> @inbounds(J[j, k] * S1[j, k]), i, T) / hξ : zero(T)) +
        (useeta(p) ? dminus_int(j -> @inbounds(Jx[i, j] * S1[i, j] + S2[i, j]), k, T) / hη : zero(T))
    V1[i, k] += dt * iρJ[i, k] * d
end
@inline Base.@propagate_inbounds function stress_sh!(B, Bd, Bu, vc, dt, ::Val{F}, i, k) where {F}
    xint, px, pz = flags(F)
    (; V1, hξ, hη, ox, nxg) = B
    nz = size(V1, 2)
    T = eltype(V1)
    w = T(SBP_W[1])
    g = j -> @inbounds(V1[j, k])
    c = (xint ? dplus_int(g, i, T) : dplus(q -> g(q - ox), i + ox, nxg)) / hξ
    e = dplus(j -> @inbounds(V1[i, j]), k, nz) / hη
    k == nz && Bu !== nothing && B.fluid == Bu.fluid && (e -= (V1[i, k] - Bu.V1[i, 1]) / (2w * hη))
    k == 1 && Bd !== nothing && B.fluid == Bd.fluid && (e -= (Bd.V1[i, size(Bd.V1, 2)] - V1[i, k]) / (2w * hη))
    p = part(B)
    usexi(p) || (c = zero(T))
    useeta(p) || (e = zero(T))
    P = pmlof(B)
    px && P !== nothing && (c = pml_stretch(pml_xmem(P, Val(:xa), Val(:xab), i, i + ox, k)..., pml_xgamma(P, i + ox, k, dt), c, dt))
    pz && P !== nothing && (e = pml_stretch(pml_zmem(P, Val(:zb), Val(:zbs), i, i + ox, k)..., pml_zgamma(P, i + ox, k, dt), e, dt))
    stress_update_sh!(B, vc, dt, i, k, c, e)
end
@inline Base.@propagate_inbounds function stress_sh!(B, Bd, Bu, vc, dt, ::Val{:int}, i, k)
    (; V1, hξ, hη) = B
    T = eltype(V1)
    p = part(B)
    c = usexi(p) ? dplus_int(j -> @inbounds(V1[j, k]), i, T) / hξ : zero(T)
    e = useeta(p) ? dplus_int(j -> @inbounds(V1[i, j]), k, T) / hη : zero(T)
    stress_update_sh!(B, vc, dt, i, k, c, e)
end
"The SH stress update from Dξ vy, Dη vy: ∂x vy = Dξ + (Jx/J) Dη, ∂z vy = Dη / J."
@inline Base.@propagate_inbounds function stress_update_sh!(B, vc, dt, i, k, c, e)
    (; S1, S2, J, Jx, mu, mu66) = B
    I = CartesianIndex(i, k)
    exy, eyz = c + Jx[I] / J[I] * e, e / J[I]
    d1, d2 = sh_increment(tilted(B), B, vc, dt, I, exy, eyz)
    S1[I] += d1
    S2[I] += d2
end
"The SH stress increments (σxy, σyz)·dt: σxy with C66, σyz with C44, in the frame of the symmetry axis if tilted (TTI)."
@inline Base.@propagate_inbounds function sh_increment(::Val{false}, B, vc, dt, I, exy, eyz)
    ts = orzero(B.ts, I)
    ts1 = unrelaxed(ts, vc)
    m6, m = B.mu66[I], B.mu[I]
    R1 = cmemvar!(memory(B, Val(:R1)), I, m6 * ts * exy, vc, dt)
    R2 = cmemvar!(memory(B, Val(:R2)), I, m * ts * eyz, vc, dt)
    return ((m6 * ts1 * exy + R1) * dt, (m * ts1 * eyz + R2) * dt)
end
@inline Base.@propagate_inbounds function sh_increment(::Val{true}, B, vc, dt, I, exy, eyz)
    s, c = sincos(B.tilt[I])                     # (the shear strains (x'y, z'y) rotate as a vector in the x-z plane)
    d1, d2 = sh_increment(Val(false), B, vc, dt, I, c * exy - s * eyz, s * exy + c * eyz)
    return (c * d1 + s * d2, c * d2 - s * d1)
end
"Sources of SH: forces fy on vy, moment tensors (Mxy, Myz) on (σxy, σyz)."
Base.@propagate_inbounds function force_sh_cv!(blocks, S, it, dt, u)
    for m in S.seg[u]:S.seg[u+1]-1
        B = blocks[S.b[m]]
        i, k, n = S.i[m], S.k[m], S.n[m]
        B.V1[i, k] += S.F[n, 1] * dt * S.c[m] * S.s[n, it] * B.iρJ[i, k]
    end
end
Base.@propagate_inbounds function moment_sh_cv!(blocks, S, it, u)
    for m in S.seg[u]:S.seg[u+1]-1
        B = blocks[S.b[m]]
        i, k, n = S.i[m], S.k[m], S.n[m]
        s = S.c[m] * S.s[n, it]
        B.S1[i, k] -= S.M[n, 1] * s
        B.S2[i, k] -= S.M[n, 2] * s
    end
end
"SH stations: vy (centred in time), uy, σyz, σxy and the strains εyz = σyz/(2C44), εxy = σxy/(2C66) (as the staggered order)."
Base.@propagate_inbounds vprev_sh_cv!(prev, blocks, St, n) = (prev[1, n] = sample_cv(blocks[St.b[n]].V1, St, n))
Base.@propagate_inbounds function record_sh_cv!(rec, acc, prev, blocks, St, itw, dt, n)
    B = blocks[St.b[n]]
    v = sample_cv(B.V1, St, n)
    if itw > 0
        sxy, syz = sample_cv(B.S1, St, n), sample_cv(B.S2, St, n)
        m, m6 = sample_cv(B.mu, St, n), sample_cv(B.mu66, St, n)
        q = ((prev[1, n] + v) / 2, acc[1, n], syz, sxy, iszero(m) ? m : syz / (2m), iszero(m6) ? m6 : sxy / (2m6))
        for j in 1:6
            rec[itw, j, n] = q[j]
        end
    end
    acc[1, n] += v * dt
end

"The stress update from Dξ v1, Dη v1, Dξ v2, Dη v2 (as the staggered solver: generalized Zener body if `vc !== nothing`)."
@inline Base.@propagate_inbounds function stress_update_cv!(B, vc, dt, i, k, a, b, c, e)
    I = CartesianIndex(i, k)
    J = B.J[I]
    q = B.Jx[I] / J
    stress_update_cv!(anisotropic(B), B, vc, dt, I, a + q * b, e / J, b / J + c + q * e)
end
@inline Base.@propagate_inbounds function stress_update_cv!(::Val{false}, B, vc, dt, I, exx, ezz, exz)
    (; S11, S22, S12, lam, mu) = B
    mu2 = 2mu[I]
    l2m = lam[I] + mu2
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    div = exx + ezz
    Rx = cmemvar!(memory(B, Val(:R11)), I, l2m * tp * div - mu2 * ts * ezz, vc, dt)
    Rz = cmemvar!(memory(B, Val(:R22)), I, l2m * tp * div - mu2 * ts * exx, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R12)), I, mu[I] * ts * exz, vc, dt)
    S11[I] += (l2m * tp1 * div - mu2 * ts1 * ezz + Rx) * dt
    S22[I] += (l2m * tp1 * div - mu2 * ts1 * exx + Rz) * dt
    S12[I] += (mu[I] * ts1 * exz + Rxz) * dt
end
# VTI (anisotropy.jl): normal stresses w (w·e) relaxing with τp and the rest (s13) with τs; σxz with C55 (`mu`)
@inline Base.@propagate_inbounds function stress_update_cv!(::Val{true}, B, vc, dt, I, exx, ezz, exz)
    d = vti_increment_cv(tilted(B), B, vc, dt, I, exx, ezz, exz)
    B.S11[I] += d[1]
    B.S22[I] += d[2]
    B.S12[I] += d[3]
end
"The VTI stress increments (σxx, σzz, σxz)·dt; tilted (TTI): in the frame of the symmetry axis (anisotropy.jl)."
@inline Base.@propagate_inbounds function vti_increment_cv(::Val{false}, B, vc, dt, I, exx, ezz, exz)
    (; wx, wz, s13, mu) = B
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    w1, w3, s = wx[I], wz[I], s13[I]
    pe = w1 * exx + w3 * ezz
    Rx = cmemvar!(memory(B, Val(:R11)), I, w1 * pe * tp + s * ts * ezz, vc, dt)
    Rz = cmemvar!(memory(B, Val(:R22)), I, w3 * pe * tp + s * ts * exx, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R12)), I, mu[I] * ts * exz, vc, dt)
    return ((w1 * pe * tp1 + s * ts1 * ezz + Rx) * dt, (w3 * pe * tp1 + s * ts1 * exx + Rz) * dt, (mu[I] * ts1 * exz + Rxz) * dt)
end
@inline Base.@propagate_inbounds function vti_increment_cv(::Val{true}, B, vc, dt, I, exx, ezz, exz)
    s, c = sincos(B.tilt[I])
    a, b, h = rot2(c, s, exx, ezz, exz / 2)                  # the strain rate in the frame of the axis (exz: engineering)
    d = vti_increment_cv(Val(false), B, vc, dt, I, a, b, 2h)
    return rot2(c, -s, d[1], d[2], d[3])
end

# Sources (lists of the nodes they reach: block, node, weight, source; sorted by node, `seg[u]:seg[u+1]-1` the entries of
# node `u`): forces on the velocities after the momentum update, moment tensors (stress glut) on the stresses after the
# stress update. One thread per node (sources may share nodes); per node in the order of the sources.
"""
The source entries `t` (NamedTuple of vectors: b, i, [j,] k, ...) sorted by node (stable), with `seg`: the first entry of
every node and one past the last.
"""
function by_node(t)
    key = :j in keys(t) ? collect(zip(t.b, t.i, t.j, t.k)) : collect(zip(t.b, t.i, t.k))
    p = sortperm(key; alg=Base.Sort.DEFAULT_STABLE)
    seg = Int32[[m for m in eachindex(p) if m == 1 || key[p[m]] != key[p[m-1]]]; length(p) + 1]
    return merge(map(v -> v[p], t), (; seg))
end
Base.@propagate_inbounds function force_cv!(blocks, S, it, dt, u)
    for m in S.seg[u]:S.seg[u+1]-1
        B = blocks[S.b[m]]
        i, k, n = S.i[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it] * B.iρJ[i, k]
        B.V1[i, k] += S.F[n, 1] * s
        B.V2[i, k] += S.F[n, 2] * s
    end
end
Base.@propagate_inbounds function moment_cv!(blocks, S, it, u)
    for m in S.seg[u]:S.seg[u+1]-1
        B = blocks[S.b[m]]
        i, k, n = S.i[m], S.k[m], S.n[m]
        s = S.c[m] * S.s[n, it]
        B.S11[i, k] -= S.M[n, 1] * s
        B.S22[i, k] -= S.M[n, 2] * s
        B.S12[i, k] -= S.M[n, 3] * s
    end
end

# Stations: Lagrange interpolation (6 × 6 nodes of their block).
Base.@propagate_inbounds function sample_cv(A, St, n)
    s = zero(eltype(A))
    for c in 1:6, a in 1:6
        s += St.wx[a, n] * St.wz[c, n] * A[St.i0[n]+a-1, St.k0[n]+c-1]
    end
    return s
end
"v^{n-1/2} at the stations (before the momentum update of a recording step)."
Base.@propagate_inbounds function vprev_cv!(prev, blocks, St, n)
    B = blocks[St.b[n]]
    prev[1, n] = sample_cv(B.V1, St, n)
    prev[2, n] = sample_cv(B.V2, St, n)
end
"After the momentum update: record v(tₙ) = (v^{n-1/2} + v^{n+1/2})/2, u(tₙ), σⁿ and the strain (from σ, unrelaxed moduli) at the
recording steps; integrate the displacement u += v^{n+1/2} dt."
Base.@propagate_inbounds function record_cv!(rec, acc, prev, blocks, St, itw, dt, n)
    B = blocks[St.b[n]]
    v1, v2 = sample_cv(B.V1, St, n), sample_cv(B.V2, St, n)
    if itw > 0
        s11, s22, s12 = sample_cv(B.S11, St, n), sample_cv(B.S22, St, n), sample_cv(B.S12, St, n)
        l, m = sample_cv(B.lam, St, n), sample_cv(B.mu, St, n)
        d = 4m * (l + m)
        # (fluid: the volumetric strain σ/λ, split equally; no shear strain)
        e = iszero(m) ? ((s11 + s22) / 4l, (s11 + s22) / 4l, m) : (((l + 2m) * s11 - l * s22) / d, ((l + 2m) * s22 - l * s11) / d, s12 / (2m))
        q = ((prev[1, n] + v1) / 2, (prev[2, n] + v2) / 2, acc[1, n], acc[2, n], s11, s22, s12, e...)
        for j in 1:10
            rec[itw, j, n] = q[j]
        end
    end
    acc[1, n] += v1 * dt
    acc[2, n] += v2 * dt
end

# ------------------------------------------------------------------ the solver

"Lagrange weights of the `np` nodes of the grid `g` around `x0`: first index and weights."
function lagrange_weights(g, x0; np=6)
    i0 = clamp(searchsortedlast(g, x0) - np ÷ 2 + 1, 1, length(g) - np + 1)
    return i0, [prod((x0 - g[m]) / (g[j] - g[m]) for m in i0:i0+np-1 if m != j) for j in i0:i0+np-1]
end
"""
The Cerjan sponge factor (as OpenSWPC: exp(-α (1 - d/na)²), α = 0.09, per step) of the node `i` of `n` (sides: both ends;
`n = typemax(Int)`: the bottom only) at the node distance d = i - 1 or n - i from the boundary.
"""
function cerjan_factor(i, n, na)
    d = min(i - 1, n - i)
    return d < na ? exp(-0.09 * (1 - d / na)^2) : 1.0
end
"A discrete delta at `x0` on the grid `g` with the norm `H`: Kaiser-windowed sinc where the grid is uniform around it,
otherwise Lagrange weights (moments up to degree 5). Returns (nodes, weights / H)."
function delta_weights(g, H, h, x0)
    c = clamp(searchsortedlast(g, x0), 1, length(g) - 1)
    pos = c + (x0 - g[c]) / (g[c+1] - g[c])
    idx = max(1, ceil(Int, pos - SINC_R)):min(length(g), floor(Int, pos + SINC_R))
    if first(idx) > SBP_NP + 1 && last(idx) < length(g) - SBP_NP
        return collect(idx), [kaiser_sinc(pos - j) / H[j] for j in idx]
    end
    i0, w = lagrange_weights(g, x0)
    return collect(i0:i0+5), [w[a] / H[i0+a-1] for a in 1:6]
end

# ------------------------------------------------------------------ MPI: subdomains in x (and y in 3D); columns stay whole

const CV_HALO = 4                       # the reach of the interior stencils
const CV_WINDOW = 8                     # host columns beyond the nodes of a process (≥ CV_HALO + the stencil half-width)
"The nodes of part `c` (1-based) of `p` of `n` nodes."
part_range(n, p, c) = fld((c - 1) * n, p)+1:fld(c * n, p)

"MPI for the curvilinear solver: `nothing` on one process, else the Cartesian communicator, rank, dims, coordinates (1-based)
and neighbours (left, right) of the `N` decomposed directions."
function cv_mpi(nproc, N)
    dims = ntuple(a -> a <= length(nproc) ? Int(nproc[a]) : 1, N)
    prod(dims) == 1 && return nothing
    MPI.Initialized() || MPI.Init()
    np = MPI.Comm_size(MPI.COMM_WORLD)
    np == prod(dims) || error("nproc = $nproc needs $(prod(dims)) MPI processes, julia runs on $np")
    comm = MPI.Cart_create(MPI.COMM_WORLD, collect(dims); periodic=fill(false, N), reorder=false)
    me = MPI.Comm_rank(comm)
    coords = Tuple(Int.(MPI.Cart_coords(comm, me))) .+ 1
    nb = ntuple(a -> MPI.Cart_shift(comm, a - 1, 1), N)
    return (; comm, me, dims, coords, nb)
end

"Exchange the halos (width CV_HALO) of the decomposed directions of `arrays` (all the same size) with the neighbours."
function cv_exchange!(par, arrays)
    par === nothing && return nothing
    nd, h = ndims(arrays[1]), CV_HALO
    for a in 1:length(par.dims)
        left, right = par.nb[a]
        n = size(arrays[1], a)
        sel(r) = ntuple(b -> b == a ? r : Colon(), nd)
        pack(r) = reduce(vcat, (vec(Array(view(A, sel(r)...))) for A in arrays))    # one message per neighbour
        sl, sr = pack(h+1:2h), pack(n-2h+1:n-h)
        rl, rr = similar(sl), similar(sr)
        MPI.Waitall([MPI.Irecv!(rl, par.comm; source=left, tag=0), MPI.Irecv!(rr, par.comm; source=right, tag=1),
                     MPI.Isend(sl, par.comm; dest=left, tag=1), MPI.Isend(sr, par.comm; dest=right, tag=0)])
        m = length(sl) ÷ length(arrays)
        for (q, A) in enumerate(arrays), (buf, r, nbr) in ((rl, 1:h, left), (rr, n-h+1:n, right))
            nbr == MPI.PROC_NULL && continue
            dest = view(A, sel(r)...)
            d = similar(A, length(dest))
            copyto!(d, view(buf, (q-1)*m+1:q*m))
            dest .= reshape(d, size(dest))
        end
    end
    return nothing
end
cv_sum(par, x) = par === nothing ? x : MPI.Allreduce(x, +, par.comm)
cv_max(par, x) = par === nothing ? x : MPI.Allreduce(x, max, par.comm)

"""
    simulate_curvilinear(model; tmax or nt, sources, ...)

2D P-SV or SH (`wave`) on a deformed (terrain-following) `CartData` (see [`curvilinear_mesh`](@ref)); called by `simulate` for such a
model. Keywords as in `simulate`, and: `dt = nothing` (from the spectral radius of the operator: 90 % of the leapfrog
limit), `tmax` (duration) instead of `nt`, `absorber = :impedance` (or `:pml`: also a PML of `na = 20` nodes at the sides
and the bottom, `:mpml`: the multiaxial PML with `mpml_ratio = 0.1`, `:cerjan`: a sponge of `na` nodes, or `:none`: free
sides and bottom), `jump = 0.05`
(relative jump of vp or vs that makes a grid row an interface, see `find_blocks`), `nproc` MPI processes along x (2D) or
`(npx, npy)` (3D): start julia with `mpiexec`; the columns are not split.
"""
function simulate_curvilinear(model::Union{CartData,DistributedModel}; dt=nothing, nt=nothing, tmax=nothing, sources, stations=Station[], stf=:kupper,
                              tbeg=0.0, absorber=:impedance, na=20, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                              density_unit=:kg_m3, snapshot_every=0, snapshot_dir="snapshots", callback=nothing, snapshots=nothing, checkpoint=nothing, checkpoint_every=0,
                              restart=false, backend=CPU(),
                              T::Type{FT}=Float32, verbose=true, wavefield=model isa CartData, jump=0.05, wave=:psv, interpolation=:sinc,
                              free_surface=:openswpc, nproc=1, driver=nothing, mpml_ratio=0.1, kwargs...) where {FT}
    isempty(kwargs) || error("simulate on a curvilinear grid: not (yet) supported: $(join(keys(kwargs), ", "))")
    wave in (:psv, :sh) || error("curvilinear grids: wave = :psv or :sh")
    sh = wave === :sh                                         # SH: vy (V1), σxy (S1), σyz (S2)
    absorber in (:impedance, :pml, :mpml, :cerjan, :none) ||
        error("curvilinear grids: absorber = :impedance, :pml, :mpml (multiaxial PML), :cerjan (with a sponge) or :none")
    absorber in (:pml, :mpml) && na < SBP_NB && error("curvilinear grids: the PML needs na >= $SBP_NB nodes (the boundary closures), got na = $na")
    absorber === :mpml && !(0 <= mpml_ratio <= 1) && error("mpml_ratio must be between 0 and 1, got $mpml_ratio")
    srcs = sources isa AbstractVector ? sources : [sources]
    all(s -> s isa Union{MomentTensorSource,ForceSource,TeleseismicWave}, srcs) ||
        error("curvilinear grids: MomentTensorSource, ForceSource and TeleseismicWave (PlaneWave: use TeleseismicWave)")
    tws = filter(s -> s isa TeleseismicWave, srcs)
    length(tws) <= 1 || error("curvilinear grids: one TeleseismicWave at a time")
    allsrcs, srcs = srcs, filter(s -> !(s isa TeleseismicWave), srcs)  # (point sources)
    wall = time()
    dev(A) = (B = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)
    # the columns of the mesh (SBP nodes)
    if model isa CartData
        X = model.x.val
        nx, nzt = size(X, 1), size(X, 3)
        all(k -> view(X, :, 1, k) == view(X, :, 1, 1), 1:nzt) || error("curvilinear grid: the columns of the mesh must be vertical")
        gξ = 1e3 .* X[:, 1, 1]
    else
        nx, nzt = length(model.x), model.nzc
        gξ = 1e3 .* model.x
    end
    gx, hξ = sbp_grid(gξ[1], gξ[end], nx)
    maximum(abs, gx .- gξ) < 1e-6 * (gξ[end] - gξ[1]) ||
        error("curvilinear grid: x must be the nodes of the SBP grid (make the mesh with curvilinear_mesh, or with GMG's curvilinear_grid(sbp_nodes(...), ...; spacing = :sbp))")
    # MPI: this process holds the nodes ia:ib in x and CV_HALO halo nodes on either side (local = global - ox)
    par = cv_mpi(nproc, 1)
    me = par === nothing ? 0 : par.me
    ia, ib = par === nothing ? (1, nx) : extrema(part_range(nx, par.dims[1], par.coords[1]))
    par === nothing || ib - ia + 1 >= SBP_NB + CV_HALO || error("curvilinear grid: too many processes in x for $nx nodes")
    hx = par === nothing ? 0 : CV_HALO
    ox = ia - hx - 1
    xr = clamp.(ia-hx:ib+hx, 1, nx)                   # the local nodes (beyond the domain ends: copies, not used)
    nl, no = length(xr), ib - ia + 1
    # the mesh and the material of the columns wa:wb (host arrays, index - ow): the nodes of this process and CV_WINDOW more
    # on either side (the metric of the halo nodes, sources and stations near the subdomain)
    wa, wb = max(1, ia - CV_WINDOW), min(nx, ib + CV_WINDOW)
    ow, nw = wa - 1, wb - wa + 1
    Zw, f = model_columns(model, wa:wb, 1:1, 2)
    Zm = 1e3 .* Zw
    all(>(0), diff(Zm; dims=2)) || error("curvilinear grid: z must increase with the row index (and rows must not touch)")
    haskey(f, :rho) && haskey(f, :vp) && haskey(f, :vs) || error("the model needs fields rho, vp and vs")
    ρ = f.rho .* (density_unit === :g_cm3 ? 1e3 : 1.0)
    vp, vs = 1e3 .* f.vp, 1e3 .* f.vs
    minimum(vp) > 0 && minimum(vs) >= 0 || error("curvilinear grids: vp > 0 and vs >= 0 (vs = 0: fluid) everywhere")
    qp = haskey(f, :Qp) ? Float64.(f.Qp) : fill(Inf, nw, nzt)
    qs = haskey(f, :Qs) ? Float64.(f.Qs) : fill(Inf, nw, nzt)
    attenuation = model isa CartData ? any(k -> haskey(model.fields, k) && any(isfinite, model.fields[k]), (:Qp, :Qs)) : model.attenuation
    aniso = has_anisotropy(model)                               # VTI: Thomsen ε, δ, γ
    thomsen(k) = haskey(f, k) ? Float64.(f[k]) : zeros(nw, nzt)
    ε, δ, γ = thomsen(:epsilon), thomsen(:delta), thomsen(:gamma)
    tti = haskey(f, :tilt)                                      # TTI: the symmetry axis tilted in the x-z plane
    haskey(f, :azimuth) && error("2D: the symmetry axis tilts in the x-z plane; give a signed `tilt` (positive towards +x), no `azimuth`")
    tti && !aniso && error("the DistributedModel returns a `tilt`: create it with `anisotropy = true`")
    θ = tti ? deg2rad.(Float64.(f.tilt)) : zeros(nw, nzt)
    ts = relaxation_times(fq_min, fq_max)
    zeta = constq_zeta(fq_min, fq_max, ts)
    # blocks: split at the interfaces; at an interface row each block takes the material of its side
    rows, below = find_blocks(Zm, vp, vs; jump, own=ia-ow:ib-ow, red=block_reducer(par))
    edges = [1; rows; nzt]
    any(b -> edges[b+1] - edges[b] + 1 < SBP_NMIN, 1:length(edges)-1) &&
        error("curvilinear grid: every block (between interfaces) needs at least $SBP_NMIN rows; interfaces at rows $rows")
    Hξ = sbp_weights(nx, hξ)
    hostblocks = map(1:length(edges)-1) do b
        r = edges[b]:edges[b+1]
        m = map(A -> copy(A[:, r]), (ρ, vp, vs, qp, qs, ε, δ, γ, θ))
        kmap = [r[k] for i in 1:nw, k in eachindex(r)]       # the model row of the material of each node (gradients)
        for (j, side, src) in ((b - 1, false, 2), (b, true, length(r) - 1))
            1 <= j <= length(rows) || continue
            kk = side ? length(r) : 1
            for i in 1:nw
                below[j][i] == side && continue
                foreach(A -> A[i, kk] = A[i, src], m)
                kmap[i, kk] = kmap[i, src]
            end
        end
        bρ, bvp, bvs, bqp, bqs, bε, bδ, bγ, bθ = m
        nz = length(r)
        _, hη = sbp_grid(0.0, 1.0, nz)
        Zb = Zm[:, r]
        J = [dminus(j -> Zb[i, j], k, nz) / hη for i in 1:nw, k in 1:nz]
        # (global stencils: at the outer nodes of the window wrong (clamped), not used)
        Jx = [-dminus(j -> Zb[clamp(j - ow, 1, nw), k], i + ow, nx) / hξ for i in 1:nw, k in 1:nz]
        minimum(J) > 0 || error("curvilinear grid: the mesh folds in block $b")
        taup, taus = NM * zeta ./ bqp, @. ifelse(bvs == 0, 0.0, NM * zeta / bqs)      # (fluid: no shear)
        mu = @. bρ * bvs^2 / visco_chi((ts,), taus, fq_ref)^2
        lam = @. bρ * bvp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu
        # the moduli in the VTI form (also of isotropic media: the adjoint)
        V = vti_moduli.(bρ, bvp, bvs, bε, bδ, bγ, visco_chi.((ts,), taup, fq_ref), visco_chi.((ts,), taus, fq_ref))
        wx, wz, s13, mu55, mu66 = ntuple(q -> getindex.(V, q), 5)
        # impedance damping at the sides and (bottom block) the bottom: a = |J·| Z / (H ρ J) with the speeds of normal
        # incidence (VTI: horizontal qP vp √(1+2ε)), and its derivatives (vp, vs, ε, γ) per component (gradients)
        a1, a2 = zeros(nw, nz), zeros(nw, nz)
        da = ntuple(_ -> ntuple(_ -> zeros(nw, nz), 4), sh ? 1 : 2)
        if absorber !== :none && sh                    # SH: σxy at the sides (C66: vs √(1+2γ)), σyz at the bottom (vs)
            for ig in (1, nx)
                wa <= ig <= wb || continue
                i, h = ig - ow, 1 / Hξ[ig]
                @. a1[i, :] += bvs[i, :] * sqrt(1 + 2bγ[i, :]) * h
                @. da[1][2][i, :] += sqrt(1 + 2bγ[i, :]) * h; @. da[1][4][i, :] += bvs[i, :] / sqrt(1 + 2bγ[i, :]) * h
            end
            if b == 1
                q = @. sqrt(Jx[:, 1]^2 + 1) / (SBP_W[1] * hη * J[:, 1])
                @. a1[:, 1] += q * bvs[:, 1]
                da[1][2][:, 1] .+= q
            end
        elseif absorber !== :none
            for ig in (1, nx)
                wa <= ig <= wb || continue
                i, h = ig - ow, 1 / Hξ[ig]
                @. a1[i, :] += bvp[i, :] * sqrt(1 + 2bε[i, :]) * h; @. a2[i, :] += bvs[i, :] * h
                @. da[1][1][i, :] += sqrt(1 + 2bε[i, :]) * h; @. da[1][3][i, :] += bvp[i, :] / sqrt(1 + 2bε[i, :]) * h
                da[2][2][i, :] .+= h
            end
            if b == 1
                Hη1 = SBP_W[1] * hη
                q = @. sqrt(Jx[:, 1]^2 + 1) / (Hη1 * J[:, 1])
                @. a1[:, 1] += q * bvs[:, 1]; @. a2[:, 1] += q * bvp[:, 1]
                da[1][2][:, 1] .+= q; da[2][1][:, 1] .+= q
            end
        end
        (; nz, hη, Z=Zb, J, Jx, ρ=bρ, lam, mu, taup, taus, a1, a2, vp=bvp, vs=bvs, ε=bε, δ=bδ, γ=bγ, wx, wz, s13, mu55, mu66, θ=bθ, kmap, da,
         fluid=block_fluid(bvs[ia-ow:ib-ow, :], b, par))
    end
    nb = length(hostblocks)
    # PML: profiles (identity until the time step is known) and memory variables (side layers: 2na columns; bottom: na rows)
    pml, mp = absorber in (:pml, :mpml), absorber === :mpml
    if pml
        nx >= 2na + 2SBP_NB + 1 || error("curvilinear grid: the PML (na = $na nodes) needs more than $(2na + 2SBP_NB) nodes in x")
        hostblocks[1].nz >= na + SBP_NB + 1 || error("curvilinear grid: the PML (na = $na nodes) needs more rows in the bottom block ($(hostblocks[1].nz))")
    end
    ident = mp ? FT.((0, 0, 1, 0)) : ntuple(q -> FT(isodd(q)), 4)   # (no damping until the time step is known)
    blocks = Tuple(map(enumerate(hostblocks)) do (bi, H)
        z() = KernelAbstractions.zeros(backend, FT, nl, H.nz)
        zs(n...) = KernelAbstractions.zeros(backend, FT, n...)
        nzp = bi == 1 ? na : 0
        player = !pml ? (;) : sh ? (; pml=(; gx=dev(fill(ident, nx)), gz=dev(fill(ident, H.nz)), np=na, nxg=nx, xv1=zs(2na, H.nz),
                                             xa=zs(2na, H.nz), zv1=zs(nl, nzp), zb=zs(nl, nzp),
                                             (mp ? (; xv1b=zs(nl, nzp), xab=zs(nl, nzp), zv1s=zs(2na, H.nz), zbs=zs(2na, H.nz),
                                                    mpml=FT(mpml_ratio), mdt=zs(1)) : (;))...)) :
                  (; pml=(; gx=dev(fill(ident, nx)), gz=dev(fill(ident, H.nz)), np=na, nxg=nx, xv1=zs(2na, H.nz), xv2=zs(2na, H.nz),
                            xa=zs(2na, H.nz), xc=zs(2na, H.nz), zv1=zs(nl, nzp), zv2=zs(nl, nzp), zb=zs(nl, nzp), ze=zs(nl, nzp),
                            (mp ? (; xv1b=zs(nl, nzp), xv2b=zs(nl, nzp), xab=zs(nl, nzp), xcb=zs(nl, nzp), zv1s=zs(2na, H.nz),
                                   zv2s=zs(2na, H.nz), zbs=zs(2na, H.nz), zes=zs(2na, H.nz), mpml=FT(mpml_ratio), mdt=zs(1)) : (;))...))
        loc(A) = dev(FT.(A[xr .- ow, :]))
        mem = attenuation ? (; R11=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz), R22=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz),
                               R12=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz)) : (;)
        vti = aniso ? (; wx=loc(H.wx), wz=loc(H.wz), s13=loc(H.s13)) : (;)
        tiltf = tti ? (; tilt=loc(H.θ)) : (;)
        r0 = H.kmap[1, 2] - 2                                       # (the mesh row of node k: r0 + k)
        sponge = absorber === :cerjan ? (; gs=loc([cerjan_factor(i + ow, nx, na) * cerjan_factor(r0 + k, typemax(Int), na) for i in 1:nw, k in 1:H.nz])) : (;)
        sh && return merge((; V1=z(), S1=z(), S2=z(), J=loc(H.J), Jx=loc(H.Jx), iρJ=loc(1 ./ (H.ρ .* H.J)), mu=loc(H.mu55), mu66=loc(H.mu66),
                            ts=attenuation ? loc(H.taus) : nothing, a1=loc(H.a1), hξ=FT(hξ), hη=FT(H.hη), ox, nxg=nx, fluid=H.fluid), tiltf,
                           attenuation ? (; R1=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz), R2=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz)) : (;),
                           sponge, player)
        merge((; V1=z(), V2=z(), S11=z(), S22=z(), S12=z(), J=loc(H.J), Jx=loc(H.Jx), iρJ=loc(1 ./ (H.ρ .* H.J)), lam=loc(H.lam),
               mu=loc(aniso ? H.mu55 : H.mu), tp=attenuation ? loc(H.taup) : nothing, ts=attenuation ? loc(H.taus) : nothing, a1=loc(H.a1),
               a2=loc(H.a2), hξ=FT(hξ), hη=FT(H.hη), ox, nxg=nx, fluid=H.fluid), mem, vti, tiltf, sponge, player)
    end)
    lite = map(B -> merge(B, (; pml=nothing)), blocks)        # (kernels without the PML: Metal allows 4 kB of arguments)
    # (kernels that see all blocks: only the arrays they read)
    recB, forceB, momB, updB, velB = slim(blocks, CV_RECORD), slim(blocks, CV_FORCE), slim(blocks, CV_STRESS), slim(blocks, CV_UPDATE), slim(blocks, CV_VEL)
    tfB = slim(blocks, CV_TF)
    below_(b) = b > 1 ? blocks[b-1] : nothing
    above_(b) = b < nb ? blocks[b+1] : nothing
    # the regions of a block, each with its kernel variant: the interior (branch-free), the top and bottom rows away from
    # the sides (closures in η only: vectorize) and the side columns (closures in both directions). All regions of a block
    # in one parallel loop on the CPU (one launch per kernel and block: many small launches cost much on a loaded machine).
    # (with MPI: the global segmentation in x, intersected with the nodes of this process)
    # PML: the columns of the side layers and the rows of the bottom layer are regions of their own (flag (xint, px, pz)).
    m = SBP_NB
    np = pml ? na : m
    regions = map(enumerate(hostblocks)) do (bi, H)
        nz = H.nz
        npz = pml && bi == 1 ? na : 0
        R = Any[]
        xsegs = ((1, m, false, pml), (m + 1, np, true, pml), (np + 1, nx - np, true, false), (nx - np + 1, nx - m, true, pml),
                 (nx - m + 1, nx, false, pml))
        zsegs = ((1, m, :edge, npz > 0), (m + 1, npz, :mid, true), (max(npz, m) + 1, nz - m, :mid, false), (nz - m + 1, nz, :edge, false))
        for (lo, hi, xint, px) in xsegs
            a, b = max(lo, ia), min(hi, ib)
            a > b && continue
            l0, w = a - ox - 1, b - a + 1
            for (k1, k2, kind, pz) in zsegs
                k2 < k1 && continue
                qx, qz = mp ? (px || pz, px || pz) : (px, pz)  # (M-PML: both directions in all layers)
                flag = !xint ? Val((false, qx, qz)) : (qx || qz) ? Val((true, qx, qz)) : kind === :mid ? Val(:int) : Val(true)
                push!(R, ((l0, k1 - 1), (w, k2 - k1 + 1), flag))
            end
        end
        Tuple(R)
    end
    # (on `blks`: the blocks, or blocks of other arrays with the same mesh: the operators of the adjoint)
    mom_k, str_k, nbr = sh ? (momentum_sh!, stress_sh!, neighbour_sh) : (momentum_cv!, stress_cv!, neighbour)
    momentum!(dtT, damp, blks=blocks) = for b in 1:nb
        foreach_region!(backend, mom_k, (blks[b], nbr(b > 1 ? blks[b-1] : nothing), nbr(b < nb ? blks[b+1] : nothing), dtT, damp), regions[b])
    end
    stress!(vc, dtT, blks=blocks) = for b in 1:nb
        foreach_region!(backend, str_k, (blks[b], nbr(b > 1 ? blks[b-1] : nothing), nbr(b < nb ? blks[b+1] : nothing), vc, dtT), regions[b])
    end
    vel(B) = sh ? (B.V1,) : (B.V1, B.V2)                       # the velocities and stresses of a block
    str(B) = sh ? (B.S1, B.S2) : (B.S11, B.S22, B.S12)
    exchange_v!() = foreach(B -> cv_exchange!(par, vel(B)), blocks)
    exchange_s!() = foreach(B -> cv_exchange!(par, str(B)), blocks)
    owned(A) = view(A, hx+1:hx+no, :)
    # the time step: power iteration for the largest eigenvalue λ of v -> M(S(v)); leapfrog is stable for dt < 2/√λ
    if dt === nothing
        rng = Random.Xoshiro(20261126 + me)                 # (seeded: the same time step in every run)
        foreach(B -> foreach(V -> copyto!(V, FT.(randn(rng, size(V)))), vel(B)), blocks)
        exchange_v!()
        λ = 0.0
        for _ in 1:100
            foreach(B -> foreach(A -> fill!(A, 0), str(B)), blocks)
            stress!(nothing, one(FT))
            exchange_s!()
            nrm0 = sqrt(cv_sum(par, value(sum(B -> sum(V -> sum(abs2, owned(V)), vel(B)), blocks))))
            foreach(B -> foreach(V -> fill!(V, 0), vel(B)), blocks)
            momentum!(one(FT), false)
            KernelAbstractions.synchronize(backend)
            exchange_v!()
            nrm = sqrt(cv_sum(par, value(sum(B -> sum(V -> sum(abs2, owned(V)), vel(B)), blocks))))
            λ = nrm / nrm0
            foreach(B -> foreach(V -> V ./= FT(nrm), vel(B)), blocks)
        end
        foreach(B -> foreach(a -> fill!(a, 0), (vel(B)..., str(B)...)), blocks)
        dt = 0.9 * 2 / sqrt(λ)
    end
    nt = something(nt, tmax === nothing ? nothing : round(Int, tmax / dt))
    nt === nothing && error("give the duration tmax (or nt, with dt)")
    dtT = FT(dt)
    vc = attenuation ? memory_coefficients(ts, dt, FT) : nothing
    if pml                      # the profiles of the time step (as OpenSWPC: 6 km/s, fcut the highest source frequency)
        fcut = maximum(s -> 1 / s.trise, allsrcs)
        xs = (gx .- gx[1]) ./ 1e3
        # (M-PML: the profiles (d, α, 1/κ, d), also of the rows above the bottom layer: no damping, α of beyond it)
        prof(x, h, L) = mp ? (c = cfs_profile(x, h, L, na, fcut); FT.((c[1], c[2], 1 / c[3], c[1]))) : FT.(damping_profile(x, h, L, na, dt, fcut))
        copyto!.(getfield.(getfield.(blocks, :pml), :gx), Ref([prof(x, xs[na+1], xs[end]) for x in xs]))
        H1 = hostblocks[1]
        # (the heights of the rows of the first column: the same on every process, exactly)
        zrow = cv_sum(par, ia == 1 ? (H1.Z[1, :] .- H1.Z[1, 1]) ./ 1e3 : zeros(H1.nz))
        copyto!(blocks[1].pml.gz, [prof(z, zrow[na+1], 1e9) for z in zrow])
        mp && foreach(b -> fill!(blocks[b].pml.gz, prof(1e8, 1.0, 1e9)), 2:nb)
        mp && foreach(B -> fill!(B.pml.mdt, dtT), blocks)
    end
    # locate points: block and η̂ (Lagrange interpolation of the mesh columns, bisection)
    function locate(x, z)
        i0, w = lagrange_weights(gx, x)
        (wa <= i0 && i0 + 5 <= wb) || error("the point ($(x / 1e3), $(z / 1e3)) km is outside the columns of process $me")
        for (b, H) in enumerate(hostblocks)
            col = [sum(w[a] * H.Z[i0+a-1-ow, k] for a in 1:6) for k in 1:H.nz]
            (col[1] - 1e-9 <= z <= col[end] + 1e-9) || continue
            gη, _ = sbp_grid(0.0, 1.0, H.nz)
            zof(η) = let (j0, wj) = lagrange_weights(gη, η)
                sum(wj[c] * col[j0+c-1] for c in 1:6)
            end
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                zof(mid) < z ? (lo = mid) : (hi = mid)
            end
            return b, (lo + hi) / 2
        end
        error("the point ($(x / 1e3), $(z / 1e3)) km is outside the model")
    end
    # sources: the nodes they reach in their block
    # (`δ`: shifted by δ [km]: the derivatives of the weights for the gradient with respect to the location)
    function source_table(list, coef, δ=(0.0, 0.0))
        tab = (b=Int[], i=Int[], k=Int[], c=Float64[], n=Int[])
        for (n, s) in enumerate(list)
            x, z = s.x + δ[1], s.z + δ[2]
            iξ, wξ = delta_weights(gx, Hξ, hξ, 1e3x)
            any(i -> ia <= i <= ib, iξ) || continue             # (no node of this process)
            b, η = locate(1e3x, 1e3z)
            H = hostblocks[b]
            gη, hη = sbp_grid(0.0, 1.0, H.nz)
            iη, wη = delta_weights(gη, sbp_weights(H.nz, hη), hη, η)
            for (a, i) in enumerate(iξ), (c, k) in enumerate(iη)
                push!(tab.b, b); push!(tab.i, i); push!(tab.k, k); push!(tab.c, wξ[a] * wη[c] * coef(H, i, k)); push!(tab.n, n)
            end
        end
        return tab
    end
    # teleseismic wave: total-field / scattered-field corrections at the nodes near the box boundary
    column(ig, jg) = (cz = model_columns(model, ig:ig, 1:1, 2); fz(k, q) = haskey(cz[2], k) ? Float64(cz[2][k][1, q]) : 0.0;
                      (vec(1e3 .* cz[1]), [(Float64(cz[2].rho[1, q]) * (density_unit === :g_cm3 ? 1e3 : 1.0), 1e3cz[2].vp[1, q], 1e3cz[2].vs[1, q],
                                            fz(:epsilon, q), fz(:delta, q), fz(:gamma, q), deg2rad(fz(:tilt, q)), 0.0) for q in axes(cz[1], 2)]))
    TW, ntw = isempty(tws) ? (nothing, 0) :
              teleseismic_cv(tws[1], 2, sh, (; hostblocks, gx, nx, ia, ib, ow, ox, dev, FT, dt, nt, tbeg, stf, blocks, na, pml, column, aniso,
                                              rows, loc=A -> A[xr .- ow, :]))[1:2]
    forces = filter(s -> s isa ForceSource, srcs)
    moments = filter(s -> s isa MomentTensorSource, srcs)
    # the nodes of this process, in local indices
    mine(t) = (q = findall(i -> ia <= i <= ib, t.i); by_node((b=t.b[q], i=t.i[q] .- ox, k=t.k[q], c=t.c[q], n=t.n[q])))
    fcoef, mcoef = (H, i, k) -> 1.0, (H, i, k) -> 1 / H.J[i-ow, k]
    tables(δ) = (mine(source_table(forces, fcoef, δ)), mine(source_table(moments, mcoef, δ)))
    ft, mt = tables((0.0, 0.0))
    # the amplitudes per source and step, on the device in windows of steps (source_window.jl)
    amp_f(n) = (s = forces[n]; amp_row(source_stf(s, stf), 1, 1, 1, tbeg, 1, dt, s.t0, s.trise))
    amp_m(n) = (s = moments[n]; amp_row(source_stf(s, stf), 1, dt, 1, tbeg, 0.5, dt, s.t0, s.trise))
    Wf, Wm = source_window(amp_f, length(forces), nt, dev, FT, backend), source_window(amp_m, length(moments), nt, dev, FT, backend)
    FS = (; b=dev(ft.b), i=dev(ft.i), k=dev(ft.k), c=dev(FT.(ft.c)), n=dev(ft.n), seg=dev(ft.seg),
          F=dev(FT[s.F[c] for s in forces, c in (sh ? (2,) : (1, 3))]),
          s=Wf.s)
    MS = (; b=dev(mt.b), i=dev(mt.i), k=dev(mt.k), c=dev(FT.(mt.c)), n=dev(mt.n), seg=dev(mt.seg),
          M=dev(FT[s.M0 * s.M[c] for s in moments, c in (sh ? (6, 4) : (1, 3, 5))]),        # (SH: σxy, σyz)
          s=Wm.s)
    # stations
    nst = length(stations)
    st = map(stations) do s
        i0, wx = lagrange_weights(gx, 1e3s.x)
        if s.surface
            b = nb
            η = 1.0
        elseif i0 + 5 >= ia && i0 <= ib                         # (other stations: not recorded on this process)
            b, η = locate(1e3s.x, 1e3s.z)
        else
            b, η = nb, 1.0
        end
        gη, _ = sbp_grid(0.0, 1.0, hostblocks[b].nz)
        k0, wz = lagrange_weights(gη, η)
        (b, i0, k0, wx, wz)
    end
    st_all = st
    own = findall(s -> ia <= s[2] + 2 <= ib, st)        # the stations of this process (the centre of their stencil)
    st = st[own]
    nst = length(st)
    St = (; b=dev([s[1] for s in st]), i0=dev([s[2] - ox for s in st]), k0=dev([s[3] for s in st]),
          wx=dev(FT[s[4][a] for a in 1:6, s in st]), wz=dev(FT[s[5][a] for a in 1:6, s in st]))
    nrec = cld(nt, record_every)
    ncol = sh ? 6 : 10                                          # (v, u, stress, strain)
    rec = KernelAbstractions.zeros(backend, FT, nrec, ncol, nst)
    acc = KernelAbstractions.zeros(backend, FT, sh ? 1 : 2, nst)
    prev = KernelAbstractions.zeros(backend, FT, sh ? 1 : 2, nst)
    # the wavefield on the mesh (the shared interface rows from the block below)
    # (with MPI: gathered on rank 0, `nothing` elsewhere)
    function velocity_field()
        cols = [k == 1 ? (1:H.nz) : (2:H.nz) for (k, H) in enumerate(hostblocks)]
        v1 = reduce(hcat, [Array(owned(blocks[b].V1))[:, cols[b]] for b in 1:nb])
        v2 = sh ? zero(v1) : reduce(hcat, [Array(owned(blocks[b].V2))[:, cols[b]] for b in 1:nb])
        zz = Zw[ia-ow:ib-ow, :]
        if par !== nothing
            parts = MPI.gather((ia, ib, v1, v2, zz), par.comm; root=0)
            me == 0 || return nothing
            v1, v2, zz = zeros(nx, nzt), zeros(nx, nzt), zeros(nx, nzt)
            for (a, b, p1, p2, pz) in parts
                v1[a:b, :] .= p1; v2[a:b, :] .= p2; zz[a:b, :] .= pz
            end
        end
        r(A) = reshape(Float64.(A), nx, 1, nzt)
        Xg = [gx[i] / 1e3 for i in 1:nx, _ in 1:1, _ in 1:nzt]
        return CartData(Xg, zero(Xg), r(zz), (; V=sh ? (zero(r(v1)), r(v1), zero(r(v1))) : (r(v1), zero(r(v1)), r(v2))))
    end
    planes = snapshots === nothing ? nothing :
             setup_planes_cv(snapshots, sh ? 1 : 2, (; hostblocks, nx, ny=1, gx, gy=nothing, Zm, ia, ib, ja=1, jb=1, oy=0, ox, dev, backend, FT, dir=snapshot_dir,
                                              me, parallel=par !== nothing, par, ow, owy=0))
    sh && planes !== nothing && :ps in snapshots.quantities && error("SH snapshots have no :ps")
    Dps, divrot! = planes !== nothing && :ps in snapshots.quantities ? divrot_cv(blocks, stress!, 2, backend, FT) : (nothing, nothing)
    movie = (snapshot_every > 0 && callback === nothing && snapshots === nothing && me == 0) ?
            movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    verbose &= me == 0
    if verbose
        @printf("SeismicWavePC: PSV on a curvilinear grid, %d × %d nodes in %d block(s) (interfaces at rows %s), %d steps of %.4g s%s%s\n",
                nx, nzt, nb, isempty(rows) ? "–" : join(rows, ", "), nt, dt, attenuation ? ", attenuation" : "",
                absorber === :none ? "" : absorber === :cerjan ? ", absorbing sides and bottom (sponge of $na nodes)" : ", absorbing sides and bottom")
    end
    # one step: v^{n+1/2} = v^{n-1/2} + dt M(σⁿ, tₙ), σ^{n+1} = σⁿ + dt S(v^{n+1/2}) (n = it - 1); recording if `record`
    function step!(it, record)
        n = it - 1
        itw = record && n % record_every == 0 ? n ÷ record_every + 1 : 0
        itw > 0 && nst > 0 && foreach_item!(backend, sh ? vprev_sh_cv! : vprev_cv!, (prev, recB, St), nst)
        momentum!(dtT, true)
        ntw > 0 && foreach_item!(backend, sh ? tf_momentum_sh! : tf_momentum_psv!, (tfB, TW, FT(tbeg + n * dt), dtT), ntw)
        absorber === :cerjan && foreach(B -> foreach(V -> V .*= B.gs, vel(B)), blocks)
        length(FS.seg) > 1 && foreach_item!(backend, sh ? force_sh_cv! : force_cv!, (forceB, FS, window_step!(Wf, it), dtT), length(FS.seg) - 1)
        par === nothing || KernelAbstractions.synchronize(backend)     # (the halos: MPI reads finished fields)
        exchange_v!()
        record && nst > 0 && foreach_item!(backend, sh ? record_sh_cv! : record_cv!, (rec, acc, prev, recB, St, itw, dtT), nst)
        stress!(vc, dtT)
        ntw > 0 && foreach_item!(backend, sh ? tf_stress_sh! : tf_stress_psv!, (updB, TW, vc, FT(tbeg + (n + 0.5) * dt), dtT), ntw)
        absorber === :cerjan && foreach(B -> foreach(S -> S .*= B.gs, str(B)), blocks)
        length(MS.seg) > 1 && foreach_item!(backend, sh ? moment_sh_cv! : moment_cv!, (momB, MS, window_step!(Wm, it)), length(MS.seg) - 1)
        par === nothing || KernelAbstractions.synchronize(backend)
        exchange_s!()
    end
    "the seismograms of all stations (velocity, displacement, stress, strain) on every process (`part`: of Born numbers)"
    function all_traces(part=identity)
        traces = zeros(nrec, ncol, length(stations))
        traces[:, :, own] .= part.(Array(rec))
        par === nothing || MPI.Allreduce!(traces, +, par.comm)
        return traces
    end
    seismograms(traces, wf=nothing) = sh ?
        Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), traces[:, 1:1, :], traces[:, 2:2, :], traces[:, 3:4, :],
                    traces[:, 5:6, :], collect(stations), (:y,), (:yz, :xy), wf) :
        Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), traces[:, 1:2, :], traces[:, 3:4, :],
                    traces[:, 5:7, :], traces[:, 8:10, :], collect(stations), (:x, :z), (:xx, :zz, :xz), wf)
    # a driver (misfit gradients) takes over
    driver === nothing || return drive_curvilinear(driver, (; backend, FT, blocks, hostblocks, nb, nx, nzt, nl, no, hx, ox, xr, ia, ib,
        par, me, dt, dtT, nt, nrec, record_every, momentum!, stress!, step!, exchange_v!, exchange_s!, owned, St, nst, FS, forces,
        Hξ, hξ, tbeg, attenuation, dev, model, all_traces, stations, density_unit, vc, own_stations=own, sponge=absorber === :cerjan, nsrc=length(allsrcs), st_all,
        seismograms, ts, fq_ref, loc=A -> A[xr .- ow, :], ftab=ft, mtab=mt, MS, moments, srcs=allsrcs, stf, tables, ow, owy=0, Wf, Wm, amp_f, amp_m,
        TW, ntw, tfkind=sh ? Val(:sh) : Val(:psv)))
    # checkpoints (one file per process): the wavefield and memory variables, the recording and the plane accumulators
    ckp = checkpoint === nothing ? nothing : par !== nothing ? "$(checkpoint)_rank$me.jls" : "$checkpoint.jls"
    restart && (ckp === nothing || !isfile(ckp)) && error("restart = true needs the checkpoint file $ckp")
    fields_of(B) = (Tuple(getfield(B, k) for k in keys(B) if occursin(r"^[VSR][0-9]+$", string(k)))..., pml_arrays(B)...)
    state = (Iterators.flatten(map(fields_of, blocks))..., rec, acc, prev,
             (planes === nothing ? () : Iterators.flatten((p.U, p.PK) for p in planes))...)
    it0 = restart ? load_state!(state, ckp, nt) + 1 : 1
    restart && verbose && @printf("  restart from step %d\n", it0 - 1)
    for it in it0:nt
        n = it - 1
        step!(it, true)
        if ckp !== nothing && checkpoint_every > 0 && it % checkpoint_every == 0 && it < nt
            KernelAbstractions.synchronize(backend)
            save_state(ckp, it, nt, state)
        end
        if planes !== nothing                          # netCDF planes: displacement and peaks every step
            store = snapshot_every > 0 && n % snapshot_every == 0
            store && Dps !== nothing && divrot!()
            for p in planes
                isempty(p.own) || foreach_item!(backend, plane_cv!, (p.B, p.U, p.PK, velB, store ? Dps : nothing, p.pts, dtT, store, Val(sh ? 1 : 2)),
                                                length(p.own))
            end
            if store
                KernelAbstractions.synchronize(backend)
                write_planes(planes, sh ? :sh : :psv, n ÷ snapshot_every + 1, tbeg + n * dt, 1.0, par !== nothing, me)
            end
        end
        if snapshot_every > 0 && n % snapshot_every == 0 && (planes === nothing || callback !== nothing)
            data = velocity_field()
            if me == 0
                callback === nothing ? (movie = write_paraview(data, "wavefield_$(lpad(n, 6, '0'))"; directory=snapshot_dir, pvd=movie,
                                                                time=tbeg + n * dt, verbose=false)) : callback(data, tbeg + n * dt)
            end
        end
        if (verbose || par !== nothing) && (it % max(1, nt ÷ 10) == 0 || it == nt)
            vmax = cv_max(par, Float64(maximum(B -> maximum(V -> maximum(abs, owned(V)), vel(B)), blocks)))
            isfinite(vmax) || error("curvilinear grid: the simulation became unstable at step $it")
            verbose && @printf("  step %6d/%d  t = %8.4f s  max|v| = %.3e m/s\n", it, nt, tbeg + it * dt, vmax)
        end
    end
    KernelAbstractions.synchronize(backend)
    movie === nothing || movie_paraview(movie)
    planes === nothing || write_peaks(planes, snapshots.quantities, sh ? :sh : :psv, 1.0, par !== nothing, me)
    verbose && @printf("  finished in %.2f s\n", time() - wall)
    return seismograms(all_traces(), wavefield ? velocity_field() : nothing)
end
