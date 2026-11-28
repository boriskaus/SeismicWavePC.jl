# Teleseismic plane waves on curvilinear grids (TeleseismicWave): total-field / scattered-field injection along a box, as on
# the staggered grid (injection.jl): the incoming field of the horizontally layered background (exact, frequency domain) is
# tabulated on a fine grid of depths and delay times; at every node whose stencils reach across the box boundary the
# updates are corrected with D(χ f) - χ D f (χ = 1 inside the box), f the incoming stresses (momentum: J σ along ξ,
# Jx σ + σ along η) or velocities (stress update, through the constitutive law: the box boundary must be elastic).
# The box: columns `i1:i2` (and `j1:j2`), mesh rows from `kB` up to the free surface.

"""
The setup of a TeleseismicWave on a curvilinear grid: tables (z up, SI) and the nodes to correct. `c`: the setup of the
simulation (N = 2 (P-SV, SH) or 3, the hostblocks of this process (window of columns), node coordinates, ...).
"""
function teleseismic_cv(w::TeleseismicWave, N, sh, c)
    (; hostblocks, gx, nx, ia, ib, ow, ox, dev, FT, dt, nt, tbeg, stf, blocks, na, pml) = c
    gy, ny, ja, jb, owy, oy = N == 3 ? (c.gy, c.ny, c.ja, c.jb, c.owy, c.oy) : (nothing, 1, 1, 1, 0, 0)
    c.aniso && error("TeleseismicWave: isotropic models only (the incoming field of VTI layers: not yet)")
    sh && w.wave !== :sh && error("TeleseismicWave: SH simulations need wave = :sh")
    !sh && N == 2 && w.wave === :sh && error("TeleseismicWave: P-SV simulations need wave = :p or :sv")
    length(w.box) == (N == 3 ? 5 : 3) || error("TeleseismicWave: box needs $(N == 3 ? 5 : 3) values in $(N)D")
    nb = length(hostblocks)
    r0 = [H.kmap[CartesianIndex(ntuple(_ -> 1, N - 1)..., 2)] - 2 for H in hostblocks]     # mesh row of node k: r0 + k
    # the box in nodes: columns with x in [xmin, xmax] (y), the row nearest to zmin at the first column
    cols(g, a, b) = (findfirst(>=(1e3a - 1e-6), g), findlast(<=(1e3b + 1e-6), g))
    i1, i2 = cols(gx, w.box[1], w.box[2])
    j1, j2 = N == 3 ? cols(gy, w.box[3], w.box[4]) : (1, 1)
    margin = SBP_NB + 5 + (pml ? na : 0)
    (i1 > margin && i2 <= nx - margin) || error("TeleseismicWave: the box must keep $margin nodes from the sides (PML and closures)")
    N == 3 && !(j1 > margin && j2 <= ny - margin) && error("TeleseismicWave: the box must keep $margin nodes from the sides in y")
    # the background: the column of the model 5 nodes outside the box (read by every process from the model)
    ir, jr = i1 - 5, N == 3 ? j1 - 5 : 1
    Zcol, mat = c.column(ir, jr)                       # heights [m] and (ρ, vp, vs) per global row of that column
    zs = Zcol[end]                                     # the free surface of the background (flat along the box)
    kB = argmin(abs.(Zcol .- 1e3w.box[end]))
    # the block of the bottom rows: kB ± 5 inside one block, away from the closures and the bottom PML
    bB = findfirst(b -> r0[b] + 1 <= kB <= r0[b] + hostblocks[b].nz, 1:nb)
    kl = kB - r0[bB]
    (kl - 5 > SBP_NB + (pml && bB == 1 ? na : 0) && kl + 5 <= hostblocks[bB].nz - SBP_NB) ||
        error("TeleseismicWave: the bottom of the box must be 5 nodes inside a block (away from interfaces, the bottom, the PML)")
    # layers of the background: a change between two rows lies at the interface row if one is (blocks), else halfway
    depth = zs .- Zcol
    tops, ρs, λs, μs = Float64[], Float64[], Float64[], Float64[]
    for q in length(depth):-1:1                        # from the surface down
        ρ, vp, vs = mat[q]
        λ, μ = ρ * (vp^2 - 2vs^2), ρ * vs^2
        if isempty(ρs) || !(ρ ≈ ρs[end] && λ ≈ λs[end] && μ ≈ μs[end])
            top = isempty(ρs) ? 0.0 : q + 1 in c.rows ? depth[q+1] : q in c.rows ? depth[q] : (depth[q] + depth[q+1]) / 2
            push!(tops, top); push!(ρs, ρ); push!(λs, λ); push!(μs, μ)
        end
    end
    L = Layers(tops, ρs, λs, μs)
    # the depth grid of the tables: the depths of the rows of this column (the nodes along the box lie on them where the rows
    # are flat), each interval split in 4 (rows that are not flat: interpolated), to one interval below the bottom. (A
    # uniform grid at the finest spacing, that of the SBP closure at the surface, needs GB-sized tables.)
    dmax = zs - Zcol[1]                                 # (to the bottom: the rows need not be flat)
    hz = minimum(diff(Zcol[max(kB - 8, 1):end]))
    dr = unique(sort(zs .- Zcol))
    zt = [dr[1]; [dr[q] + s / 4 * (dr[q+1] - dr[q]) for q in 1:length(dr)-1 for s in 1:4]]
    push!(zt, 2zt[end] - zt[end-1])
    # (one-sided columns at the layer boundaries: σxx jumps; a node on an interface row takes the side of its block)
    nreg = length(zt)
    zt = [zt; [z + q * 1e-6 for z in L.top[2:end] for q in (-1, 1)]]
    # direction, slowness, delay (as injection_setup)
    N == 2 && abs(sind(w.backazimuth)) < 0.99 && error("TeleseismicWave: in 2D the backazimuth must be 90° or 270°")
    e3 = N == 2 ? (sign(-sind(w.backazimuth)), 0.0) : (-sind(w.backazimuth), -cosd(w.backazimuth))
    vel(j) = w.wave === :p ? sqrt((L.lam[j] + 2L.mu[j]) / L.rho[j]) : sqrt(L.mu[j] / L.rho[j])
    p = sind(w.angle) / vel(length(L.top))
    pv = ntuple(a -> p * e3[a], Val(N - 1))
    ηs = [real(vslow(vel(j), p)) for j in eachindex(L.top)]
    # (t0: when the wave reaches the box as given, its deepest corner on the source side; not the nodes next to it)
    zB = zs - 1e3w.box[end]
    D = vertical_delay(L, ηs, zB)
    lo, hi = (1e3w.box[1], N == 3 ? 1e3w.box[3] : 0.0), (1e3w.box[2], N == 3 ? 1e3w.box[4] : 0.0)
    x_up = ntuple(a -> pv[a] >= 0 ? lo[a] : hi[a], Val(N - 1))
    tshift = w.t0 - sum(pv .* x_up) - D
    ext = N == 3 ? ((gx[1], gx[end]), (gy[1], gy[end])) : ((gx[1], gx[end]),)
    pmin = sum(ntuple(a -> min(pv[a] * ext[a][1], pv[a] * ext[a][2]), Val(N - 1)))
    pmax = sum(ntuple(a -> max(pv[a] * ext[a][1], pv[a] * ext[a][2]), Val(N - 1)))
    τ0 = min(tbeg - tshift - pmax - 2dt, -(vertical_delay(L, ηs, 0.0) - vertical_delay(L, ηs, dmax)) - 2w.trise)
    τ1 = tbeg + (nt + 1) * dt - tshift - pmin + 2dt
    dτ = table_step(w, stf, τ0, τ1, dt)
    fk = w.wave === :sh ? Val(:sh) : Val(:psv)
    e = N == 2 ? (e3[1], 0.0) : e3
    θ = N == 2 ? (0.0, e3[1]) : (-e3[2], e3[1])
    rot = rotation(fk, e, θ)
    names = sh ? (:Vy, :Syz, :Sxy) : N == 2 ? (:Vx, :Vz, :Sxx, :Szz, :Sxz) : (:Vx, :Vy, :Vz, :Sxx, :Syy, :Szz, :Syz, :Sxz, :Sxy)
    tabs = teleseismic_tables_cv(w, fk, rot, p, L, zt, names, τ0, τ1, dτ, stf, FT)
    # z up: vz, σxz, σyz change sign (the tables: z down)
    flip = map(n -> n in (:Vz, :Sxz, :Syz) ? -one(FT) : one(FT), NamedTuple{names}(names))
    # the nodes to correct (owned): a stencil (±4) reaches across the box boundary
    inside(ig, jg, kg) = i1 <= ig <= i2 && j1 <= jg <= j2 && kg >= kB
    band = (b=Int32[], i=Int32[], j=Int32[], k=Int32[])
    for (b, H) in enumerate(hostblocks), k in 1:H.nz, jg in (N == 3 ? (ja:jb) : (1:1)), ig in ia:ib
        kg = r0[b] + k
        χ = inside(ig, jg, kg)
        near = any(q -> inside(ig + q, jg, kg) != χ, -4:4) || (N == 3 && any(q -> inside(ig, jg + q, kg) != χ, -4:4)) ||
               (b == bB && any(q -> inside(ig, jg, kg + q) != χ, -4:4))
        near || continue
        Iw = N == 2 ? CartesianIndex(ig - ow, k) : CartesianIndex(ig - ow, jg - owy, k)
        (H.taup[Iw] == 0 && H.taus[Iw] == 0) || error("TeleseismicWave: the model must be elastic (no Qp, Qs) along the box boundary")
        # horizontally layered there: the material of the background at the depth of the node (away from layer boundaries)
        dn = zs - H.Z[Iw]
        kg == length(Zcol) && abs(dn) > 1e-3 * hz &&
            error("TeleseismicWave: the free surface must be flat along the box boundary (at the height of the column $(ir) outside it)")
        jl = max(searchsortedlast(L.top, dn), 1)          # (a node above zs: the surface row below errors)
        if all(t -> abs(dn - t) > 0.6 * hz, L.top[2:end])
            ρ, λ, μ = H.ρ[Iw], H.ρ[Iw] * (H.vp[Iw]^2 - 2H.vs[Iw]^2), H.ρ[Iw] * H.vs[Iw]^2
            (isapprox(ρ, L.rho[jl]; rtol=1e-6) && isapprox(λ, L.lam[jl]; rtol=1e-6) && isapprox(μ, L.mu[jl]; rtol=1e-6)) ||
                error("TeleseismicWave: the model must be horizontally layered along the box boundary (it differs at the node $((ig, jg, kg)))")
        end
        push!(band.b, b); push!(band.i, ig - ox); push!(band.j, jg - oy); push!(band.k, k)
    end
    # the position of every local node in the tables (column, fractional): its depth; on an interface row the one-sided
    # column of its side (the block above an interface: the upper layer)
    function position(b, H)
        zr = view(zt, 1:nreg)                            # (0-based fractional column in the regular part)
        frac(d) = (j = clamp(searchsortedlast(zr, d), 1, nreg - 1); (j - 1) + (d - zr[j]) / (zr[j+1] - zr[j]))
        P = frac.(zs .- c.loc(H.Z))
        for (q, top) in enumerate(L.top[2:end]), I in CartesianIndices(P)
            k = I[N]
            onrow = (k == 1 && b > 1) || (k == H.nz && b < nb)
            (onrow && abs(zs - c.loc(H.Z)[I] - top) < 1e-3 * hz) || continue
            P[I] = nreg + 2(q - 1) + (k == 1 ? 0 : 1)                  # (0-based column: upper side, lower side)
        end
        return dev(FT.(P))
    end
    Zn = Tuple(position(b, H) for (b, H) in enumerate(hostblocks))
    gxd = dev(FT.(gx)); gyd = N == 3 ? dev(FT.(gy)) : nothing
    W = (; tab=map(dev, tabs), flip, τ0=FT(τ0), dτ=FT(dτ), zs=FT(zs), p=FT.(pv), tshift=FT(tshift), gx=gxd, gy=gyd,
         i1=Int32(i1), i2=Int32(i2), j1=Int32(j1), j2=Int32(j2), kB=Int32(kB), bB=Int32(bB), r0=Tuple(Int32.(r0)), Zn,
         b=dev(band.b), i=dev(band.i), j=dev(band.j), k=dev(band.k))
    return W, length(band.b), (; layers=L, zs, kB, i1, i2, j1, j2, p=pv, tshift)
end

"Tables `tab[name][iτ, iz]` of the incoming field at the depths `zt` [m] and delay times `τ0 + (iτ - 1) dτ` (SI; z down)."
function teleseismic_tables_cv(w, fk::Val{m}, rot, p, L, zt, names, τ0, τ1, dτ, stf, ::Type{T}) where {m,T}
    wave = m === :sh ? :sh : w.wave
    N = nextpow(2, ceil(Int, 2(τ1 - τ0) / dτ))
    S = rfft([w.amplitude * moment_rate(stf, t, 0.0, w.trise) for t in τ0 .+ (0:N-1) .* dτ])
    ω = 2π .* (0:N÷2) ./ (N * dτ)
    keep = findall(>(1e-9 * maximum(abs, S)), abs.(S))
    G = Dict(n => zeros(ComplexF64, length(keep), length(zt)) for n in names)
    for (iq, q) in enumerate(keep)
        amp, Wl = layer_amplitudes(L, p, wave, ω[q], fk)
        for (kz, z) in enumerate(zt)
            f = layer_field(L, amp, Wl, ω[q], z, fk)
            for n in names
                v = sum((f[cc] * a for (cc, a) in rot[n]); init=0.0im)
                G[n][iq, kz] = first(string(n)) == 'V' ? S[q] * v : -S[q] * v
            end
        end
    end
    nτ = ceil(Int, (τ1 - τ0) / dτ) + 2
    return NamedTuple{names}(map(n -> time_table(G[n], keep, N, nτ, 1.0, T), names))
end

"Is the node (global column ig, jg, mesh row kg) inside the box?  Is it within 4 rows of its bottom (the η corrections)?"
@inline tf_bottom(W, b, kg) = (b == W.bB) & (abs(kg - W.kB) <= 4)
@inline tf_inside(W, ig, jg, kg) = (W.i1 <= ig <= W.i2) & (W.j1 <= jg <= W.j2) & (kg >= W.kB)
"The incoming field `name` (z up) at the local node `I` of block `b` (global horizontal position `x`) at the time `t`."
Base.@propagate_inbounds function tf_field(W, ::Val{name}, b, I, x, t) where {name}
    A = getfield(W.tab, name)
    T = eltype(A)
    d = W.Zn[b][I]                                   # (the position in the table columns)
    jz = floor(Int, d)
    u = d - jz
    (jz < 0 || jz + 1 > size(A, 2)) && return zero(T)
    jz + 2 > size(A, 2) && (u = zero(u))             # (the last column)
    f = (t - W.tshift - x - W.τ0) / W.dτ
    jt = floor(Int, f)
    (jt < 0 || jt + 2 > size(A, 1)) && return zero(T)
    a = f - jt
    jz2 = iszero(u) ? jz + 1 : jz + 2                # (a one-sided column alone)
    v = (1 - a) * ((1 - u) * A[jt+1, jz+1] + u * A[jt+1, jz2]) + a * ((1 - u) * A[jt+2, jz+1] + u * A[jt+2, jz2])
    return getfield(W.flip, name) * v
end
@inline tf_px(W, ig, jg) = W.gy === nothing ? W.p[1] * W.gx[ig] : W.p[1] * W.gx[ig] + W.p[2] * W.gy[jg]

# --- P-SV: corrections of the momentum update (incoming stresses at time t) and of the stress update (velocities)
"The corrections of the momentum update (P-SV) at band node `n`: (block, local node, the two divergences)."
Base.@propagate_inbounds function tf_div_psv(blocks, W, t, n)
    b, i, k = W.b[n], W.i[n], W.k[n]
    B = blocks[b]
    (; J, Jx, hξ, hη, ox) = B
    T = eltype(B.V1)
    r0 = W.r0[b]
    χ(ii, kk) = T(tf_inside(W, ii + ox, 1, r0 + kk))
    χn = χ(i, k)
    s(name, ii, kk) = (xp = tf_px(W, ii + ox, 1); tf_field(W, name, b, CartesianIndex(ii, kk), xp, t))
    x1 = dminus_int(j -> (χn - χ(j, k)) * J[j, k] * s(Val(:Sxx), j, k), i, T) / hξ
    x2 = dminus_int(j -> (χn - χ(j, k)) * J[j, k] * s(Val(:Sxz), j, k), i, T) / hξ
    bot = tf_bottom(W, b, r0 + k)
    z1 = bot ? dminus_int(j -> (χn - χ(i, j)) * (Jx[i, j] * s(Val(:Sxx), i, j) + s(Val(:Sxz), i, j)), k, T) / hη : zero(T)
    z2 = bot ? dminus_int(j -> (χn - χ(i, j)) * (Jx[i, j] * s(Val(:Sxz), i, j) + s(Val(:Szz), i, j)), k, T) / hη : zero(T)
    return b, CartesianIndex(i, k), (x1 + z1, x2 + z2)
end
Base.@propagate_inbounds function tf_momentum_psv!(blocks, W, t, dt, n)
    b, I, d = tf_div_psv(blocks, W, t, n)
    B = blocks[b]
    B.V1[I] += dt * B.iρJ[I] * d[1]
    B.V2[I] += dt * B.iρJ[I] * d[2]
end
"The corrections of the stress update (P-SV): (block, local node, the raw derivatives Dξ v1, Dη v1, Dξ v2, Dη v2)."
Base.@propagate_inbounds function tf_raw_psv(blocks, W, t, n)
    b, i, k = W.b[n], W.i[n], W.k[n]
    B = blocks[b]
    (; hξ, hη, ox) = B
    T = eltype(B.V1)
    r0 = W.r0[b]
    χ(ii, kk) = T(tf_inside(W, ii + ox, 1, r0 + kk))
    χn = χ(i, k)
    v(name, ii, kk) = (xp = tf_px(W, ii + ox, 1); tf_field(W, name, b, CartesianIndex(ii, kk), xp, t))
    a = dplus_int(j -> (χn - χ(j, k)) * v(Val(:Vx), j, k), i, T) / hξ
    bot = tf_bottom(W, b, r0 + k)
    bb = bot ? dplus_int(j -> (χn - χ(i, j)) * v(Val(:Vx), i, j), k, T) / hη : zero(T)
    c = dplus_int(j -> (χn - χ(j, k)) * v(Val(:Vz), j, k), i, T) / hξ
    e = bot ? dplus_int(j -> (χn - χ(i, j)) * v(Val(:Vz), i, j), k, T) / hη : zero(T)
    return b, CartesianIndex(i, k), (a, bb, c, e)
end
Base.@propagate_inbounds function tf_stress_psv!(blocks, W, vc, t, dt, n)
    b, I, r = tf_raw_psv(blocks, W, t, n)
    stress_update_cv!(blocks[b], vc, dt, I[1], I[2], r...)
end

# --- SH: vy (V1), σxy (S1), σyz (S2)
Base.@propagate_inbounds function tf_div_sh(blocks, W, t, n)
    b, i, k = W.b[n], W.i[n], W.k[n]
    B = blocks[b]
    (; J, Jx, hξ, hη, ox) = B
    T = eltype(B.V1)
    r0 = W.r0[b]
    χ(ii, kk) = T(tf_inside(W, ii + ox, 1, r0 + kk))
    χn = χ(i, k)
    s(name, ii, kk) = (xp = tf_px(W, ii + ox, 1); tf_field(W, name, b, CartesianIndex(ii, kk), xp, t))
    x1 = dminus_int(j -> (χn - χ(j, k)) * J[j, k] * s(Val(:Sxy), j, k), i, T) / hξ
    z1 = tf_bottom(W, b, r0 + k) ? dminus_int(j -> (χn - χ(i, j)) * (Jx[i, j] * s(Val(:Sxy), i, j) + s(Val(:Syz), i, j)), k, T) / hη : zero(T)
    return b, CartesianIndex(i, k), (x1 + z1,)
end
Base.@propagate_inbounds function tf_momentum_sh!(blocks, W, t, dt, n)
    b, I, d = tf_div_sh(blocks, W, t, n)
    blocks[b].V1[I] += dt * blocks[b].iρJ[I] * d[1]
end
Base.@propagate_inbounds function tf_raw_sh(blocks, W, t, n)
    b, i, k = W.b[n], W.i[n], W.k[n]
    B = blocks[b]
    (; hξ, hη, ox) = B
    T = eltype(B.V1)
    r0 = W.r0[b]
    χ(ii, kk) = T(tf_inside(W, ii + ox, 1, r0 + kk))
    χn = χ(i, k)
    v(ii, kk) = (xp = tf_px(W, ii + ox, 1); tf_field(W, Val(:Vy), b, CartesianIndex(ii, kk), xp, t))
    c = dplus_int(j -> (χn - χ(j, k)) * v(j, k), i, T) / hξ
    e = tf_bottom(W, b, r0 + k) ? dplus_int(j -> (χn - χ(i, j)) * v(i, j), k, T) / hη : zero(T)
    return b, CartesianIndex(i, k), (c, e)
end
Base.@propagate_inbounds function tf_stress_sh!(blocks, W, vc, t, dt, n)
    b, I, r = tf_raw_sh(blocks, W, t, n)
    stress_update_sh!(blocks[b], vc, dt, I[1], I[2], r...)
end

# --- 3D: σ11..σ12 = Sxx, Syy, Szz, -Syz, -Sxz, Sxy (z up: the flips are in the tables); tractions along η: Jx σc1 + Jy σc2 + σc3
Base.@propagate_inbounds function tf_div3(blocks, W, t, n)
    b, i, j, k = W.b[n], W.i[n], W.j[n], W.k[n]
    B = blocks[b]
    (; J, Jx, Jy, hξ, hψ, hη, ox, oy) = B
    T = eltype(B.V1)
    r0 = W.r0[b]
    χ(ii, jj, kk) = T(tf_inside(W, ii + ox, jj + oy, r0 + kk))
    χn = χ(i, j, k)
    s(name, ii, jj, kk) = (xp = tf_px(W, ii + ox, jj + oy); tf_field(W, name, b, CartesianIndex(ii, jj, kk), xp, t))
    σ(c, d, ii, jj, kk) = c == 1 ? (d == 1 ? s(Val(:Sxx), ii, jj, kk) : d == 2 ? s(Val(:Sxy), ii, jj, kk) : s(Val(:Sxz), ii, jj, kk)) :
                         c == 2 ? (d == 1 ? s(Val(:Sxy), ii, jj, kk) : d == 2 ? s(Val(:Syy), ii, jj, kk) : s(Val(:Syz), ii, jj, kk)) :
                                  (d == 1 ? s(Val(:Sxz), ii, jj, kk) : d == 2 ? s(Val(:Syz), ii, jj, kk) : s(Val(:Szz), ii, jj, kk))
    bot = tf_bottom(W, b, r0 + k)
    d = ntuple(Val(3)) do c
        x = dminus_int(q -> (χn - χ(q, j, k)) * J[q, j, k] * σ(c, 1, q, j, k), i, T) / hξ
        y = dminus_int(q -> (χn - χ(i, q, k)) * J[i, q, k] * σ(c, 2, i, q, k), j, T) / hψ
        z = bot ? dminus_int(q -> (χn - χ(i, j, q)) * (Jx[i, j, q] * σ(c, 1, i, j, q) + Jy[i, j, q] * σ(c, 2, i, j, q) + σ(c, 3, i, j, q)), k, T) / hη : zero(T)
        x + y + z
    end
    return b, CartesianIndex(i, j, k), d
end
Base.@propagate_inbounds function tf_momentum3!(blocks, W, t, dt, n)
    b, I, d = tf_div3(blocks, W, t, n)
    B = blocks[b]
    r = dt * B.iρJ[I]
    B.V1[I] += r * d[1]; B.V2[I] += r * d[2]; B.V3[I] += r * d[3]
end
"The strain corrections (3D): (block, node, (∂x v1, ∂y v2, ∂z v3, eyz, exz, exy) with engineering shear strains)."
Base.@propagate_inbounds function tf_strain3(blocks, W, t, n)
    b, i, j, k = W.b[n], W.i[n], W.j[n], W.k[n]
    B = blocks[b]
    (; J, Jx, Jy, hξ, hψ, hη, ox, oy) = B
    T = eltype(B.V1)
    r0 = W.r0[b]
    χ(ii, jj, kk) = T(tf_inside(W, ii + ox, jj + oy, r0 + kk))
    χn = χ(i, j, k)
    v(name, ii, jj, kk) = (xp = tf_px(W, ii + ox, jj + oy); tf_field(W, name, b, CartesianIndex(ii, jj, kk), xp, t))
    vc_(c, ii, jj, kk) = c == 1 ? v(Val(:Vx), ii, jj, kk) : c == 2 ? v(Val(:Vy), ii, jj, kk) : v(Val(:Vz), ii, jj, kk)
    bot = tf_bottom(W, b, r0 + k)
    x = ntuple(c -> dplus_int(q -> (χn - χ(q, j, k)) * vc_(c, q, j, k), i, T) / hξ, Val(3))
    y = ntuple(c -> dplus_int(q -> (χn - χ(i, q, k)) * vc_(c, i, q, k), j, T) / hψ, Val(3))
    z = ntuple(c -> bot ? dplus_int(q -> (χn - χ(i, j, q)) * vc_(c, i, j, q), k, T) / hη : zero(T), Val(3))
    I = CartesianIndex(i, j, k)
    iJ = 1 / J[I]
    qx, qy = Jx[I] * iJ, Jy[I] * iJ
    return b, I, (x[1] + qx * z[1], y[2] + qy * z[2], z[3] * iJ, z[2] * iJ + y[3] + qy * z[3], z[1] * iJ + x[3] + qx * z[3],
                  y[1] + qy * z[1] + x[2] + qx * z[2])
end
Base.@propagate_inbounds function tf_stress3!(blocks, W, vc, t, dt, n)
    b, I, e = tf_strain3(blocks, W, t, n)
    stress3_update_cv!(anisotropic(blocks[b]), blocks[b], vc, dt, I, e...)
end

# --- gradient: the corrections are sources that depend on the material at the band nodes: 1/(ρJ) (momentum) and the
# moduli in the VTI form (stress; the box boundary is elastic): dt V̄·d and dt σ̄·∂C e
Base.@propagate_inbounds function tf_grad_momentum!(G, A, blocks, W, t, dt, ::Val{kind}, n) where {kind}
    b, I, d = kind === :psv ? tf_div_psv(blocks, W, t, n) : kind === :sh ? tf_div_sh(blocks, W, t, n) : tf_div3(blocks, W, t, n)
    a = A[b]
    v = kind === :psv ? (a.V1[I], a.V2[I]) : kind === :sh ? (a.V1[I],) : (a.V1[I], a.V2[I], a.V3[I])
    G[b].iρJ[I] += dt * sum(v .* d)
end
Base.@propagate_inbounds function tf_grad_stress!(G, A, blocks, Mv, W, t, dt, ::Val{kind}, n) where {kind}
    if kind === :psv
        b, I, (ra, rb, rc, re) = tf_raw_psv(blocks, W, t, n)
        B = blocks[b]
        q = B.Jx[I] / B.J[I]
        e = (ra + q * rb, re / B.J[I], rb / B.J[I] + rc + q * re)            # exx, ezz, γxz
        P = (A[b].S11[I], A[b].S22[I], A[b].S12[I])
        wx, wz, s13, mu = Mv[b][1][I], Mv[b][2][I], Mv[b][3][I], Mv[b][4][I]
        pe, pP = wx * e[1] + wz * e[2], wx * P[1] + wz * P[2]
        g = G[b]
        g.wx[I] += dt * (pe * P[1] + e[1] * pP); g.wz[I] += dt * (pe * P[2] + e[2] * pP)
        g.s13[I] += dt * (P[1] * e[2] + P[2] * e[1]); g.mu[I] += dt * P[3] * e[3]
    elseif kind === :sh
        b, I, (rc, re) = tf_raw_sh(blocks, W, t, n)
        B = blocks[b]
        exy, eyz = rc + B.Jx[I] / B.J[I] * re, re / B.J[I]
        G[b].m66[I] += dt * A[b].S1[I] * exy
        G[b].mu[I] += dt * A[b].S2[I] * eyz
    else
        b, I, e = tf_strain3(blocks, W, t, n)
        a = A[b]
        P = (a.S11[I], a.S22[I], a.S33[I], a.S23[I], a.S13[I], a.S12[I])
        wx, wz = Mv[b][1][I], Mv[b][2][I]
        pe, pP = wx * (e[1] + e[2]) + wz * e[3], wx * (P[1] + P[2]) + wz * P[3]
        g = G[b]
        g.wx[I] += dt * (pe * (P[1] + P[2]) + (e[1] + e[2]) * pP); g.wz[I] += dt * (pe * P[3] + e[3] * pP)
        g.s13[I] += dt * (P[1] * e[3] + P[3] * e[1] + P[2] * e[3] + P[3] * e[2])
        g.m66[I] += dt * (-2(P[1] * e[2] + P[2] * e[1]) + P[6] * e[6])
        g.mu[I] += dt * (P[4] * e[4] + P[5] * e[5])
    end
end
