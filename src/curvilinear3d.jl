# 3D on a terrain-following (curvilinear) grid: the scheme of curvilinear.jl with a second horizontal direction. The mesh
# has vertical columns (x = ξ, y = ψ on SBP nodes, z = Z(ξ, ψ, η)); metric: J = z_η, Jξ = (J, 0, 0), Jψ = (0, J, 0),
# Jη = (Jx, Jy, 1) = (-z_ξ, -z_ψ, 1). Momentum: ρ J ∂t v_c = Dξ(J σ_c1) + Dψ(J σ_c2) + Dη(Jx σ_c1 + Jy σ_c2 + σ_c3); gradient:
# ∂x = Dξ + (Jx/J) Dη, ∂y = Dψ + (Jy/J) Dη, ∂z = Dη/J. SAT terms, blocks, time stepping as in 2D.

"`flag ? interior : general` SBP derivative at the local index i (global i + o of n: MPI subdomains; the boundary closures
in global indices). No closure over the element type: the kernels stay type stable."
@inline dmin(g::G, i, o, n, ::Val{int}, ::Type{T}) where {G,int,T} = int ? dminus_int(g, i, T) : dminus(q -> g(q - o), i + o, n)
@inline dpl(g::G, i, o, n, ::Val{int}, ::Type{T}) where {G,int,T} = int ? dplus_int(g, i, T) : dplus(q -> g(q - o), i + o, n)

# The tractions across η = const (scaled), G_c = Jx σ_c1 + Jy σ_c2 + σ_c3, in a pointwise pass before the momentum update:
# the η stencils then read one array each (7× faster on CPUs: 8 planes of three arrays did not stay in cache).
@inline Base.@propagate_inbounds function traction3_cv!(B, i, j, k)
    jx, jy = B.Jx[i, j, k], B.Jy[i, j, k]
    B.G1[i, j, k] = jx * B.S11[i, j, k] + jy * B.S12[i, j, k] + B.S13[i, j, k]
    B.G2[i, j, k] = jx * B.S12[i, j, k] + jy * B.S22[i, j, k] + B.S23[i, j, k]
    B.G3[i, j, k] = jx * B.S13[i, j, k] + jy * B.S23[i, j, k] + B.S33[i, j, k]
end

"The arrays of a neighbouring block that the interface terms read (small kernel arguments: Metal allows 4 kB)."
neighbour3(B) = (; V1=B.V1, V2=B.V2, V3=B.V3, G1=B.G1, G2=B.G2, G3=B.G3, fluid=B.fluid)
neighbour3(::Nothing) = nothing

# Momentum at node (i, j, k); the region flags (fx, fy, fz[, px, py, pz]): interior in x, y, η (no closures, no SAT terms
# there), and in the PML layers of x, y, the bottom. The parts in ξ, ψ, η (with their SAT terms) are summed (the adjoint
# applies them separately: `part`; the PML stretches them, see curvilinear.jl).
@inline flags3(F) = length(F) == 3 ? (F..., false, false, false) : F
@inline yslab(P, jg) = jg <= P.np ? jg : jg - P.nyg + 2P.np
# M-PML (see curvilinear.jl): the coefficients per node, d∥ the sum of the other two axes; the memory variables of a part
# outside its own layer: in the layers of the other axes (4D arrays, the component last: `x…y`, `x…z` for ξ in the y layers
# and the bottom, `y…x`, `y…z`, `z…x`, `z…y`), each node in one of them. Which one: `w`, the layers of the region (a constant of
# its kernel, the flags' 7th entry: branches on the node would blow up the kernels, beyond what Metal's compiler handles).
@inline Base.@propagate_inbounds pml_xgamma3(P, ig, jg, k) = ismpml(P) ? mpml_gamma(P, P.gx[ig], P.gy[jg][4] + P.gz[k][4], P.mdt[1]) : P.gx[ig]
@inline Base.@propagate_inbounds pml_ygamma3(P, ig, jg, k) = ismpml(P) ? mpml_gamma(P, P.gy[jg], P.gx[ig][4] + P.gz[k][4], P.mdt[1]) : P.gy[jg]
@inline Base.@propagate_inbounds pml_zgamma3(P, ig, jg, k) = ismpml(P) ? mpml_gamma(P, P.gz[k], P.gx[ig][4] + P.gy[jg][4], P.mdt[1]) : P.gz[k]
@inline layers3(F) = length(F) == 7 ? F[7] : (false, false, false)
"The stretched ξ part `d` (memory `a` in the x layers; M-PML: `ay` in the y layers, `az` in the bottom one, component `c`)."
@inline Base.@propagate_inbounds function stretch_x3(P, i, j, k, ig, jg, w, ::Val{a}, ::Val{ay}, ::Val{az}, c, γ, d, dt) where {a,ay,az}
    (!ismpml(P) || w[1]) && return pml_stretch(getfield(P, a), CartesianIndex(xslab(P, ig), j, k), γ, d, dt)
    w[2] && return pml_stretch(getfield(P, ay), CartesianIndex(i, yslab(P, jg), k, c), γ, d, dt)
    return pml_stretch(getfield(P, az), CartesianIndex(i, j, k, c), γ, d, dt)
end
"The stretched ψ part (memory `a` in the y layers; M-PML: `ax` in the x layers, `az` in the bottom one)."
@inline Base.@propagate_inbounds function stretch_y3(P, i, j, k, ig, jg, w, ::Val{a}, ::Val{ax}, ::Val{az}, c, γ, d, dt) where {a,ax,az}
    (!ismpml(P) || w[2]) && return pml_stretch(getfield(P, a), CartesianIndex(i, yslab(P, jg), k), γ, d, dt)
    w[1] && return pml_stretch(getfield(P, ax), CartesianIndex(xslab(P, ig), j, k, c), γ, d, dt)
    return pml_stretch(getfield(P, az), CartesianIndex(i, j, k, c), γ, d, dt)
end
"The stretched η part (memory `a` in the bottom layer; M-PML: `ax` in the x layers above it, `ay` in the y layers)."
@inline Base.@propagate_inbounds function stretch_z3(P, i, j, k, ig, jg, w, ::Val{a}, ::Val{ax}, ::Val{ay}, c, γ, d, dt) where {a,ax,ay}
    (!ismpml(P) || w[3]) && return pml_stretch(getfield(P, a), CartesianIndex(i, j, k), γ, d, dt)
    w[1] && return pml_stretch(getfield(P, ax), CartesianIndex(xslab(P, ig), j, k, c), γ, d, dt)
    return pml_stretch(getfield(P, ay), CartesianIndex(i, yslab(P, jg), k, c), γ, d, dt)
end
@inline Base.@propagate_inbounds function momentum3_cv!(B, Bd, Bu, dt, damp, ::Val{F}, i, j, k) where {F}
    (; V1, V2, V3, S11, S22, S33, S23, S13, S12, G1, G2, G3, J, iρJ, a1, a2, a3, hξ, hψ, hη, ox, oy, nxg, nyg) = B
    flags = flags3(F)
    fx, fy, fz = Val(flags[1]), Val(flags[2]), Val(flags[3])
    nx, ny, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    # J σ_c1 along ξ, J σ_c2 along ψ, G_c along η
    x1 = dmin(q -> @inbounds(J[q, j, k] * S11[q, j, k]), i, ox, nxg, fx, T) / hξ
    x2 = dmin(q -> @inbounds(J[q, j, k] * S12[q, j, k]), i, ox, nxg, fx, T) / hξ
    x3 = dmin(q -> @inbounds(J[q, j, k] * S13[q, j, k]), i, ox, nxg, fx, T) / hξ
    y1 = dmin(q -> @inbounds(J[i, q, k] * S12[i, q, k]), j, oy, nyg, fy, T) / hψ
    y2 = dmin(q -> @inbounds(J[i, q, k] * S22[i, q, k]), j, oy, nyg, fy, T) / hψ
    y3 = dmin(q -> @inbounds(J[i, q, k] * S23[i, q, k]), j, oy, nyg, fy, T) / hψ
    z1 = dmin(q -> @inbounds(G1[i, j, q]), k, 0, nz, fz, T) / hη
    z2 = dmin(q -> @inbounds(G2[i, j, q]), k, 0, nz, fz, T) / hη
    z3 = dmin(q -> @inbounds(G3[i, j, q]), k, 0, nz, fz, T) / hη
    if !flags[1] && (i + ox == 1 || i + ox == nxg)       # sides x: the traction J σ_c1 removed
        s = (i + ox == 1 ? 1 : -1) * J[i, j, k] / (w * hξ)
        x1 += s * S11[i, j, k]; x2 += s * S12[i, j, k]; x3 += s * S13[i, j, k]
    end
    if !flags[2] && (j + oy == 1 || j + oy == nyg)       # sides y: J σ_c2
        s = (j + oy == 1 ? 1 : -1) * J[i, j, k] / (w * hψ)
        y1 += s * S12[i, j, k]; y2 += s * S22[i, j, k]; y3 += s * S23[i, j, k]
    end
    if !flags[3] && (k == 1 || k == nz)                 # free surface, bottom, interfaces
        g1, g2, g3 = G1[i, j, k], G2[i, j, k], G3[i, j, k]
        if k == nz
            if Bu === nothing
                z1 -= g1 / (w * hη); z2 -= g2 / (w * hη); z3 -= g3 / (w * hη)
            else
                e1, e2, e3 = sat_traction(B.fluid != Bu.fluid, (B.Jx[i, j, k], B.Jy[i, j, k], one(T)), (g1, g2, g3), (Bu.G1[i, j, 1], Bu.G2[i, j, 1], Bu.G3[i, j, 1]))
                z1 -= e1 / (2w * hη); z2 -= e2 / (2w * hη); z3 -= e3 / (2w * hη)
            end
        end
        if k == 1
            if Bd === nothing
                z1 += g1 / (w * hη); z2 += g2 / (w * hη); z3 += g3 / (w * hη)
            else
                n = size(Bd.V1, 3)
                e1, e2, e3 = sat_traction(B.fluid != Bd.fluid, (B.Jx[i, j, k], B.Jy[i, j, k], one(T)), (g1, g2, g3), (Bd.G1[i, j, n], Bd.G2[i, j, n], Bd.G3[i, j, n]))
                z1 += e1 / (2w * hη); z2 += e2 / (2w * hη); z3 += e3 / (2w * hη)
            end
        end
    end
    p = part(B)
    usexi(p) || (x1 = x2 = x3 = zero(T))
    usepsi(p) || (y1 = y2 = y3 = zero(T))
    useeta(p) || (z1 = z2 = z3 = zero(T))
    P = pmlof(B)
    if flags[4] && P !== nothing
        γ, g = pml_xgamma3(P, i + ox, j + oy, k), (P, i, j, k, i + ox, j + oy, layers3(F))
        x1 = stretch_x3(g..., Val(:xv1), Val(:xvy), Val(:xvz), 1, γ, x1, dt)
        x2 = stretch_x3(g..., Val(:xv2), Val(:xvy), Val(:xvz), 2, γ, x2, dt)
        x3 = stretch_x3(g..., Val(:xv3), Val(:xvy), Val(:xvz), 3, γ, x3, dt)
    end
    if flags[5] && P !== nothing
        γ, g = pml_ygamma3(P, i + ox, j + oy, k), (P, i, j, k, i + ox, j + oy, layers3(F))
        y1 = stretch_y3(g..., Val(:yv1), Val(:yvx), Val(:yvz), 1, γ, y1, dt)
        y2 = stretch_y3(g..., Val(:yv2), Val(:yvx), Val(:yvz), 2, γ, y2, dt)
        y3 = stretch_y3(g..., Val(:yv3), Val(:yvx), Val(:yvz), 3, γ, y3, dt)
    end
    if flags[6] && P !== nothing
        γ, g = pml_zgamma3(P, i + ox, j + oy, k), (P, i, j, k, i + ox, j + oy, layers3(F))
        z1 = stretch_z3(g..., Val(:zv1), Val(:zvx), Val(:zvy), 1, γ, z1, dt)
        z2 = stretch_z3(g..., Val(:zv2), Val(:zvx), Val(:zvy), 2, γ, z2, dt)
        z3 = stretch_z3(g..., Val(:zv3), Val(:zvx), Val(:zvy), 3, γ, z3, dt)
    end
    d1, d2, d3 = x1 + y1 + z1, x2 + y2 + z2, x3 + y3 + z3
    r = iρJ[i, j, k]
    if flags[1] && flags[2] && flags[3]
        V1[i, j, k] += dt * r * d1; V2[i, j, k] += dt * r * d2; V3[i, j, k] += dt * r * d3
    else
        b1, b2, b3 = damp ? (a1[i, j, k] * dt / 2, a2[i, j, k] * dt / 2, a3[i, j, k] * dt / 2) : (zero(T), zero(T), zero(T))
        V1[i, j, k] = ((1 - b1) * V1[i, j, k] + dt * r * d1) / (1 + b1)
        V2[i, j, k] = ((1 - b2) * V2[i, j, k] + dt * r * d2) / (1 + b2)
        V3[i, j, k] = ((1 - b3) * V3[i, j, k] + dt * r * d3) / (1 + b3)
    end
end

# Stress at node (i, j, k): gradients with D₊ (half the velocity mismatch at interfaces added to Dη v), and the stress
# update of the staggered 3D solver (generalized Zener body).
@inline Base.@propagate_inbounds function stress3_cv!(B, Bd, Bu, vc, dt, ::Val{F}, i, j, k) where {F}
    (; V1, V2, V3, S11, S22, S33, S23, S13, S12, J, Jx, Jy, lam, mu, hξ, hψ, hη, ox, oy, nxg, nyg) = B
    flags = flags3(F)
    fx, fy, fz = Val(flags[1]), Val(flags[2]), Val(flags[3])
    nx, ny, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    # (no local helper closures: one capturing the type T would make the kernel allocate)
    x1 = dpl(q -> @inbounds(V1[q, j, k]), i, ox, nxg, fx, T) / hξ
    x2 = dpl(q -> @inbounds(V2[q, j, k]), i, ox, nxg, fx, T) / hξ
    x3 = dpl(q -> @inbounds(V3[q, j, k]), i, ox, nxg, fx, T) / hξ
    y1 = dpl(q -> @inbounds(V1[i, q, k]), j, oy, nyg, fy, T) / hψ
    y2 = dpl(q -> @inbounds(V2[i, q, k]), j, oy, nyg, fy, T) / hψ
    y3 = dpl(q -> @inbounds(V3[i, q, k]), j, oy, nyg, fy, T) / hψ
    z1 = dpl(q -> @inbounds(V1[i, j, q]), k, 0, nz, fz, T) / hη
    z2 = dpl(q -> @inbounds(V2[i, j, q]), k, 0, nz, fz, T) / hη
    z3 = dpl(q -> @inbounds(V3[i, j, q]), k, 0, nz, fz, T) / hη
    if !flags[3]
        if k == nz && Bu !== nothing
            e1, e2, e3 = sat_velocity(B.fluid != Bu.fluid, (Jx[i, j, k], Jy[i, j, k], one(T)), (V1[i, j, k], V2[i, j, k], V3[i, j, k]),
                                      (Bu.V1[i, j, 1], Bu.V2[i, j, 1], Bu.V3[i, j, 1]))
            z1 -= e1 / (2w * hη); z2 -= e2 / (2w * hη); z3 -= e3 / (2w * hη)
        end
        if k == 1 && Bd !== nothing
            n = size(Bd.V1, 3)
            e1, e2, e3 = sat_velocity(B.fluid != Bd.fluid, (Jx[i, j, k], Jy[i, j, k], one(T)), (V1[i, j, k], V2[i, j, k], V3[i, j, k]),
                                      (Bd.V1[i, j, n], Bd.V2[i, j, n], Bd.V3[i, j, n]))
            z1 += e1 / (2w * hη); z2 += e2 / (2w * hη); z3 += e3 / (2w * hη)
        end
    end
    p = part(B)
    usexi(p) || (x1 = x2 = x3 = zero(T))
    usepsi(p) || (y1 = y2 = y3 = zero(T))
    useeta(p) || (z1 = z2 = z3 = zero(T))
    P = pmlof(B)
    if flags[4] && P !== nothing
        γ, g = pml_xgamma3(P, i + ox, j + oy, k), (P, i, j, k, i + ox, j + oy, layers3(F))
        x1 = stretch_x3(g..., Val(:xa1), Val(:xay), Val(:xaz), 1, γ, x1, dt)
        x2 = stretch_x3(g..., Val(:xa2), Val(:xay), Val(:xaz), 2, γ, x2, dt)
        x3 = stretch_x3(g..., Val(:xa3), Val(:xay), Val(:xaz), 3, γ, x3, dt)
    end
    if flags[5] && P !== nothing
        γ, g = pml_ygamma3(P, i + ox, j + oy, k), (P, i, j, k, i + ox, j + oy, layers3(F))
        y1 = stretch_y3(g..., Val(:ya1), Val(:yax), Val(:yaz), 1, γ, y1, dt)
        y2 = stretch_y3(g..., Val(:ya2), Val(:yax), Val(:yaz), 2, γ, y2, dt)
        y3 = stretch_y3(g..., Val(:ya3), Val(:yax), Val(:yaz), 3, γ, y3, dt)
    end
    if flags[6] && P !== nothing
        γ, g = pml_zgamma3(P, i + ox, j + oy, k), (P, i, j, k, i + ox, j + oy, layers3(F))
        z1 = stretch_z3(g..., Val(:za1), Val(:zax), Val(:zay), 1, γ, z1, dt)
        z2 = stretch_z3(g..., Val(:za2), Val(:zax), Val(:zay), 2, γ, z2, dt)
        z3 = stretch_z3(g..., Val(:za3), Val(:zax), Val(:zay), 3, γ, z3, dt)
    end
    I = CartesianIndex(i, j, k)
    iJ = 1 / J[I]
    qx, qy = Jx[I] * iJ, Jy[I] * iJ
    dxV1, dyV2, dzV3 = x1 + qx * z1, y2 + qy * z2, z3 * iJ
    eyz = z2 * iJ + y3 + qy * z3
    exz = z1 * iJ + x3 + qx * z3
    exy = y1 + qy * z1 + x2 + qx * z2
    stress3_update_cv!(anisotropic(B), B, vc, dt, I, dxV1, dyV2, dzV3, eyz, exz, exy)
end
@inline Base.@propagate_inbounds function stress3_update_cv!(::Val{false}, B, vc, dt, I, dxV1, dyV2, dzV3, eyz, exz, exy)
    (; S11, S22, S33, S23, S13, S12, lam, mu) = B
    mu2 = 2mu[I]
    l2m = lam[I] + mu2
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    div = dxV1 + dyV2 + dzV3
    Rx = cmemvar!(memory(B, Val(:R11)), I, l2m * tp * div - mu2 * ts * (dyV2 + dzV3), vc, dt)
    Ry = cmemvar!(memory(B, Val(:R22)), I, l2m * tp * div - mu2 * ts * (dxV1 + dzV3), vc, dt)
    Rz = cmemvar!(memory(B, Val(:R33)), I, l2m * tp * div - mu2 * ts * (dxV1 + dyV2), vc, dt)
    S11[I] += (l2m * tp1 * div - mu2 * ts1 * (dyV2 + dzV3) + Rx) * dt
    S22[I] += (l2m * tp1 * div - mu2 * ts1 * (dxV1 + dzV3) + Ry) * dt
    S33[I] += (l2m * tp1 * div - mu2 * ts1 * (dxV1 + dyV2) + Rz) * dt
    m = mu[I]
    Ryz = cmemvar!(memory(B, Val(:R23)), I, m * ts * eyz, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R13)), I, m * ts * exz, vc, dt)
    Rxy = cmemvar!(memory(B, Val(:R12)), I, m * ts * exy, vc, dt)
    S23[I] += (m * ts1 * eyz + Ryz) * dt
    S13[I] += (m * ts1 * exz + Rxz) * dt
    S12[I] += (m * ts1 * exy + Rxy) * dt
end
# VTI: σyz with C44 = C55 (`mu`), σxy with C66 (`mu66`), C12 − C11 = −2 C66
@inline Base.@propagate_inbounds function stress3_update_cv!(::Val{true}, B, vc, dt, I, dxV1, dyV2, dzV3, eyz, exz, exy)
    d = vti3_increment_cv(tilted(B), B, vc, dt, I, dxV1, dyV2, dzV3, eyz, exz, exy)
    B.S11[I] += d[1]
    B.S22[I] += d[2]
    B.S33[I] += d[3]
    B.S23[I] += d[4]
    B.S13[I] += d[5]
    B.S12[I] += d[6]
end
"TTI: the VTI increments in the frame of the symmetry axis (anisotropy.jl), rotated back."
@inline Base.@propagate_inbounds function vti3_increment_cv(::Val{true}, B, vc, dt, I, dxV1, dyV2, dzV3, eyz, exz, exy)
    A = axis_frame(B.tilt[I], B.azim[I])
    e = rot3(A, (dxV1, dyV2, dzV3, eyz / 2, exz / 2, exy / 2))        # (shear strain rates: engineering)
    d = vti3_increment_cv(Val(false), B, vc, dt, I, e[1], e[2], e[3], 2e[4], 2e[5], 2e[6])
    return rot3(transpose3(A), d)
end
"The VTI stress increments (σxx, σyy, σzz, σyz, σxz, σxy)·dt."
@inline Base.@propagate_inbounds function vti3_increment_cv(::Val{false}, B, vc, dt, I, dxV1, dyV2, dzV3, eyz, exz, exy)
    (; wx, wz, s13, mu, mu66) = B
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    w1, w3, s, n66 = wx[I], wz[I], s13[I], -2mu66[I]
    pe = w1 * (dxV1 + dyV2) + w3 * dzV3
    ox, oy, oz = n66 * dyV2 + s * dzV3, n66 * dxV1 + s * dzV3, s * (dxV1 + dyV2)
    Rx = cmemvar!(memory(B, Val(:R11)), I, w1 * pe * tp + ox * ts, vc, dt)
    Ry = cmemvar!(memory(B, Val(:R22)), I, w1 * pe * tp + oy * ts, vc, dt)
    Rz = cmemvar!(memory(B, Val(:R33)), I, w3 * pe * tp + oz * ts, vc, dt)
    m, m6 = mu[I], mu66[I]
    Ryz = cmemvar!(memory(B, Val(:R23)), I, m * ts * eyz, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R13)), I, m * ts * exz, vc, dt)
    Rxy = cmemvar!(memory(B, Val(:R12)), I, m6 * ts * exy, vc, dt)
    return ((w1 * pe * tp1 + ox * ts1 + Rx) * dt, (w1 * pe * tp1 + oy * ts1 + Ry) * dt, (w3 * pe * tp1 + oz * ts1 + Rz) * dt,
            (m * ts1 * eyz + Ryz) * dt, (m * ts1 * exz + Rxz) * dt, (m6 * ts1 * exy + Rxy) * dt)
end

Base.@propagate_inbounds function force3_cv!(blocks, S, it, dt, u)
    for m in S.seg[u]:S.seg[u+1]-1
        B = blocks[S.b[m]]
        i, j, k, n = S.i[m], S.j[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it] * B.iρJ[i, j, k]
        B.V1[i, j, k] += S.F[n, 1] * s; B.V2[i, j, k] += S.F[n, 2] * s; B.V3[i, j, k] += S.F[n, 3] * s
    end
end
Base.@propagate_inbounds function moment3_cv!(blocks, S, it, u)
    for m in S.seg[u]:S.seg[u+1]-1
        B = blocks[S.b[m]]
        i, j, k, n = S.i[m], S.j[m], S.k[m], S.n[m]
        s = S.c[m] * S.s[n, it]
        B.S11[i, j, k] -= S.M[n, 1] * s; B.S22[i, j, k] -= S.M[n, 2] * s; B.S33[i, j, k] -= S.M[n, 3] * s
        B.S23[i, j, k] -= S.M[n, 4] * s; B.S13[i, j, k] -= S.M[n, 5] * s; B.S12[i, j, k] -= S.M[n, 6] * s
    end
end
Base.@propagate_inbounds function sample3_cv(A, St, n)
    s = zero(eltype(A))
    for c in 1:6, b in 1:6, a in 1:6
        s += St.wx[a, n] * St.wy[b, n] * St.wz[c, n] * A[St.i0[n]+a-1, St.j0[n]+b-1, St.k0[n]+c-1]
    end
    return s
end
Base.@propagate_inbounds function vprev3_cv!(prev, blocks, St, n)
    B = blocks[St.b[n]]
    prev[1, n] = sample3_cv(B.V1, St, n); prev[2, n] = sample3_cv(B.V2, St, n); prev[3, n] = sample3_cv(B.V3, St, n)
end
Base.@propagate_inbounds function record3_cv!(rec, acc, prev, blocks, St, itw, dt, n)
    B = blocks[St.b[n]]
    v = (sample3_cv(B.V1, St, n), sample3_cv(B.V2, St, n), sample3_cv(B.V3, St, n))
    if itw > 0
        σ = (sample3_cv(B.S11, St, n), sample3_cv(B.S22, St, n), sample3_cv(B.S33, St, n), sample3_cv(B.S23, St, n),
             sample3_cv(B.S13, St, n), sample3_cv(B.S12, St, n))
        l, m = sample3_cv(B.lam, St, n), sample3_cv(B.mu, St, n)
        tr = (σ[1] + σ[2] + σ[3]) * l / (3l + 2m)
        ef = (σ[1] + σ[2] + σ[3]) / 9l                 # (fluid: the volumetric strain, split equally; no shear strain)
        e = iszero(m) ? (ef, ef, ef, m, m, m) : ((σ[1] - tr) / 2m, (σ[2] - tr) / 2m, (σ[3] - tr) / 2m, σ[4] / 2m, σ[5] / 2m, σ[6] / 2m)
        for c in 1:3
            rec[itw, c, n] = (prev[c, n] + v[c]) / 2
            rec[itw, 3+c, n] = acc[c, n]
        end
        for c in 1:6
            rec[itw, 6+c, n] = σ[c]
            rec[itw, 12+c, n] = e[c]
        end
    end
    for c in 1:3
        acc[c, n] += v[c] * dt
    end
end

"""
    simulate_curvilinear3d(model; tmax or nt, sources, ...)

3D on a deformed (terrain-following) `CartData` (vertical columns; see [`curvilinear_mesh`](@ref)); called by `simulate`.
Keywords as [`SeismicWavePC.simulate_curvilinear`](@ref).
"""
function simulate_curvilinear3d(model::Union{CartData,DistributedModel}; dt=nothing, nt=nothing, tmax=nothing, sources, stations=Station[], stf=:kupper,
                                tbeg=0.0, absorber=:impedance, na=20, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                                density_unit=:kg_m3, snapshot_every=0, snapshot_dir="snapshots", callback=nothing, snapshots=nothing, checkpoint=nothing, checkpoint_every=0,
                              restart=false, backend=CPU(),
                                T::Type{FT}=Float32, verbose=true, wavefield=model isa CartData, jump=0.05, wave=:psv, interpolation=:sinc,
                                free_surface=:openswpc, nproc=1, driver=nothing, mpml_ratio=0.1, kwargs...) where {FT}
    isempty(kwargs) || error("simulate on a curvilinear grid: not (yet) supported: $(join(keys(kwargs), ", "))")
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
    if model isa CartData
        X, Y = model.x.val, model.y.val
        nx, ny, nzt = size(X)
        all(k -> view(X, :, :, k) == view(X, :, :, 1), 1:nzt) && all(k -> view(Y, :, :, k) == view(Y, :, :, 1), 1:nzt) ||
            error("curvilinear grid: the columns of the mesh must be vertical")
        all(j -> view(X, :, j, 1) == view(X, :, 1, 1), 1:ny) && all(i -> view(Y, i, :, 1) == view(Y, 1, :, 1), 1:nx) ||
            error("curvilinear grid: x must not depend on y and y not on x")
        gξ, gψ = 1e3 .* X[:, 1, 1], 1e3 .* Y[1, :, 1]
    else
        nx, ny, nzt = length(model.x), length(model.y), model.nzc
        gξ, gψ = 1e3 .* model.x, 1e3 .* model.y
    end
    gx, hξ = sbp_grid(gξ[1], gξ[end], nx)
    gy, hψ = sbp_grid(gψ[1], gψ[end], ny)
    (maximum(abs, gx .- gξ) < 1e-6 * (gξ[end] - gξ[1]) && maximum(abs, gy .- gψ) < 1e-6 * (gψ[end] - gψ[1])) ||
        error("curvilinear grid: x and y must be the nodes of the SBP grid (make the mesh with curvilinear_mesh, or with GMG's curvilinear_grid(sbp_nodes(...), sbp_nodes(...), ...; spacing = :sbp))")
    # MPI: this process holds the nodes ia:ib in x, ja:jb in y, and CV_HALO halo nodes around them (local = global - offset)
    par = cv_mpi(nproc, 2)
    me = par === nothing ? 0 : par.me
    ia, ib = par === nothing ? (1, nx) : extrema(part_range(nx, par.dims[1], par.coords[1]))
    ja, jb = par === nothing ? (1, ny) : extrema(part_range(ny, par.dims[2], par.coords[2]))
    par === nothing || min(ib - ia, jb - ja) + 1 >= SBP_NB + CV_HALO || error("curvilinear grid: too many processes for $nx × $ny nodes")
    hx = par === nothing ? 0 : CV_HALO
    ox, oy = ia - hx - 1, ja - hx - 1
    xr, yr = clamp.(ia-hx:ib+hx, 1, nx), clamp.(ja-hx:jb+hx, 1, ny)
    nlx, nly, nox, noy = length(xr), length(yr), ib - ia + 1, jb - ja + 1
    # the mesh and the material of the columns wa:wb × va:vb (host arrays, index - (ow, owy)): the nodes of this process and
    # CV_WINDOW more around them
    wa, wb, va, vb = max(1, ia - CV_WINDOW), min(nx, ib + CV_WINDOW), max(1, ja - CV_WINDOW), min(ny, jb + CV_WINDOW)
    ow, owy, nwx, nwy = wa - 1, va - 1, wb - wa + 1, vb - va + 1
    Zw, f = model_columns(model, wa:wb, va:vb, 3)
    Zm = 1e3 .* Zw
    all(>(0), diff(Zm; dims=3)) || error("curvilinear grid: z must increase with the row index (and rows must not touch)")
    haskey(f, :rho) && haskey(f, :vp) && haskey(f, :vs) || error("the model needs fields rho, vp and vs")
    ρ = f.rho .* (density_unit === :g_cm3 ? 1e3 : 1.0)
    vp, vs = 1e3 .* f.vp, 1e3 .* f.vs
    minimum(vp) > 0 && minimum(vs) >= 0 || error("curvilinear grids: vp > 0 and vs >= 0 (vs = 0: fluid) everywhere")
    qp = haskey(f, :Qp) ? Float64.(f.Qp) : fill(Inf, size(Zw))
    qs = haskey(f, :Qs) ? Float64.(f.Qs) : fill(Inf, size(Zw))
    attenuation = model isa CartData ? any(k -> haskey(model.fields, k) && any(isfinite, model.fields[k]), (:Qp, :Qs)) : model.attenuation
    aniso = has_anisotropy(model)                               # VTI: Thomsen ε, δ, γ
    thomsen(k) = haskey(f, k) ? Float64.(f[k]) : zeros(size(Zw))
    ε, δ, γ = thomsen(:epsilon), thomsen(:delta), thomsen(:gamma)
    tti = haskey(f, :tilt) || haskey(f, :azimuth)               # TTI: the tilt θ and azimuth φ of the symmetry axis
    tti && !aniso && error("the DistributedModel returns a `tilt`/`azimuth`: create it with `anisotropy = true`")
    θ, φ = (haskey(f, k) ? deg2rad.(Float64.(f[k])) : zeros(size(Zw)) for k in (:tilt, :azimuth))
    ts = relaxation_times(fq_min, fq_max)
    zeta = constq_zeta(fq_min, fq_max, ts)
    # blocks: from the columns (flattened; the decisions over the owned columns of all processes)
    flat(A) = reshape(A, nwx * nwy, nzt)
    ownc = vec([i + (j - 1) * nwx for i in ia-ow:ib-ow, j in ja-owy:jb-owy])
    rows, below = find_blocks(flat(Zm), flat(vp), flat(vs); jump, own=ownc, red=block_reducer(par))
    edges = [1; rows; nzt]
    any(b -> edges[b+1] - edges[b] + 1 < SBP_NMIN, 1:length(edges)-1) &&
        error("curvilinear grid: every block (between interfaces) needs at least $SBP_NMIN rows; interfaces at rows $rows")
    Hξ, Hψ = sbp_weights(nx, hξ), sbp_weights(ny, hψ)
    hostblocks = map(1:length(edges)-1) do b
        r = edges[b]:edges[b+1]
        m = map(A -> copy(A[:, :, r]), (ρ, vp, vs, qp, qs, ε, δ, γ, θ, φ))
        kmap = [r[k] for i in 1:nwx, j in 1:nwy, k in eachindex(r)]       # the model row of the material of each node
        for (q, side, src) in ((b - 1, false, 2), (b, true, length(r) - 1))
            1 <= q <= length(rows) || continue
            kk = side ? length(r) : 1
            for c in 1:nwx*nwy
                below[q][c] == side && continue
                i, j = mod1(c, nwx), fld1(c, nwx)
                foreach(A -> A[i, j, kk] = A[i, j, src], m)
                kmap[i, j, kk] = kmap[i, j, src]
            end
        end
        bρ, bvp, bvs, bqp, bqs, bε, bδ, bγ, bθ, bφ = m
        nz = length(r)
        _, hη = sbp_grid(0.0, 1.0, nz)
        Zb = Zm[:, :, r]
        J = [dminus(q -> Zb[i, j, q], k, nz) / hη for i in 1:nwx, j in 1:nwy, k in 1:nz]
        # (global stencils: at the outer nodes of the window wrong (clamped), not used)
        Jx = [-dminus(q -> Zb[clamp(q - ow, 1, nwx), j, k], i + ow, nx) / hξ for i in 1:nwx, j in 1:nwy, k in 1:nz]
        Jy = [-dminus(q -> Zb[i, clamp(q - owy, 1, nwy), k], j + owy, ny) / hψ for i in 1:nwx, j in 1:nwy, k in 1:nz]
        minimum(J) > 0 || error("curvilinear grid: the mesh folds in block $b")
        taup, taus = NM * zeta ./ bqp, @. ifelse(bvs == 0, 0.0, NM * zeta / bqs)      # (fluid: no shear)
        mu = @. bρ * bvs^2 / visco_chi((ts,), taus, fq_ref)^2
        lam = @. bρ * bvp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu
        V = vti_moduli.(bρ, bvp, bvs, bε, bδ, bγ, visco_chi.((ts,), taup, fq_ref), visco_chi.((ts,), taus, fq_ref))
        wx, wz, s13, mu55, mu66 = ntuple(q -> getindex.(V, q), 5)
        a1, a2, a3 = zeros(nwx, nwy, nz), zeros(nwx, nwy, nz), zeros(nwx, nwy, nz)
        # a = |J·| Z / (H ρ J) with the speeds of normal incidence: normal qP (sides: vp √(1+2ε)), tangential S (σxy: C66,
        # vs √(1+2γ)); and the derivatives (gradients): da[c] = (∂vp, ∂vs, ∂ε, ∂γ) of a_c
        da = ntuple(_ -> ntuple(_ -> zeros(nwx, nwy, nz), 4), 3)
        if absorber !== :none
            # add h × (speed) to a_c and its derivatives: kind :p (horizontal qP), :sh (C66), :s (vs), :pv (vertical qP)
            function add!(c, s, h, kind)
                A, d = (a1, a2, a3)[c], da[c]
                if kind === :p
                    @. A[s...] += h * bvp[s...] * sqrt(1 + 2bε[s...])
                    @. d[1][s...] += h * sqrt(1 + 2bε[s...]); @. d[3][s...] += h * bvp[s...] / sqrt(1 + 2bε[s...])
                elseif kind === :sh
                    @. A[s...] += h * bvs[s...] * sqrt(1 + 2bγ[s...])
                    @. d[2][s...] += h * sqrt(1 + 2bγ[s...]); @. d[4][s...] += h * bvs[s...] / sqrt(1 + 2bγ[s...])
                elseif kind === :s
                    @. A[s...] += h * bvs[s...]; @. d[2][s...] += h
                else
                    @. A[s...] += h * bvp[s...]; @. d[1][s...] += h
                end
            end
            for i in (1, nx)
                wa <= i <= wb || continue
                s = (i-ow:i-ow, :, :)
                add!(1, s, 1 / Hξ[i], :p); add!(2, s, 1 / Hξ[i], :sh); add!(3, s, 1 / Hξ[i], :s)
            end
            for j in (1, ny)
                va <= j <= vb || continue
                s = (:, j-owy:j-owy, :)
                add!(1, s, 1 / Hψ[j], :sh); add!(2, s, 1 / Hψ[j], :p); add!(3, s, 1 / Hψ[j], :s)
            end
            if b == 1
                g = @. sqrt(Jx[:, :, 1:1]^2 + Jy[:, :, 1:1]^2 + 1) / (SBP_W[1] * hη * J[:, :, 1:1])
                s = (:, :, 1:1)
                add!(1, s, g, :s); add!(2, s, g, :s); add!(3, s, g, :pv)
            end
        end
        (; nz, hη, Z=Zb, J, Jx, Jy, ρ=bρ, lam, mu, taup, taus, a1, a2, a3, vp=bvp, vs=bvs, ε=bε, δ=bδ, γ=bγ, wx, wz, s13, mu55, mu66,
           θ=bθ, φ=bφ, kmap, da, fluid=block_fluid(bvs[ia-ow:ib-ow, ja-owy:jb-owy, :], b, par))
    end
    nb = length(hostblocks)
    # PML: profiles (identity until the time step is known) and memory variables (x, y layers: 2na columns, 2na rows; bottom: na)
    pml, mp = absorber in (:pml, :mpml), absorber === :mpml
    if pml
        min(nx, ny) >= 2na + 2SBP_NB + 1 || error("curvilinear grid: the PML (na = $na nodes) needs more than $(2na + 2SBP_NB) nodes in x and y")
        hostblocks[1].nz >= na + SBP_NB + 1 || error("curvilinear grid: the PML (na = $na nodes) needs more rows in the bottom block ($(hostblocks[1].nz))")
    end
    ident = mp ? FT.((0, 0, 1, 0)) : ntuple(q -> FT(isodd(q)), 4)   # (no damping until the time step is known)
    blocks = Tuple(map(enumerate(hostblocks)) do (bi, H)
        zs(n...) = KernelAbstractions.zeros(backend, FT, n...)
        nzp = bi == 1 ? na : 0
        player = pml ? (; pml=(; gx=dev(fill(ident, nx)), gy=dev(fill(ident, ny)), gz=dev(fill(ident, H.nz)), np=na, nxg=nx, nyg=ny,
                                 xv1=zs(2na, nly, H.nz), xv2=zs(2na, nly, H.nz), xv3=zs(2na, nly, H.nz), xa1=zs(2na, nly, H.nz),
                                 xa2=zs(2na, nly, H.nz), xa3=zs(2na, nly, H.nz), yv1=zs(nlx, 2na, H.nz), yv2=zs(nlx, 2na, H.nz),
                                 yv3=zs(nlx, 2na, H.nz), ya1=zs(nlx, 2na, H.nz), ya2=zs(nlx, 2na, H.nz), ya3=zs(nlx, 2na, H.nz),
                                 zv1=zs(nlx, nly, nzp), zv2=zs(nlx, nly, nzp), zv3=zs(nlx, nly, nzp), za1=zs(nlx, nly, nzp),
                                 za2=zs(nlx, nly, nzp), za3=zs(nlx, nly, nzp),
                                 (mp ? (; xvy=zs(nlx, 2na, H.nz, 3), xvz=zs(nlx, nly, nzp, 3), yvx=zs(2na, nly, H.nz, 3), yvz=zs(nlx, nly, nzp, 3), zvx=zs(2na, nly, H.nz, 3), zvy=zs(nlx, 2na, H.nz, 3),
                                        xay=zs(nlx, 2na, H.nz, 3), xaz=zs(nlx, nly, nzp, 3), yax=zs(2na, nly, H.nz, 3), yaz=zs(nlx, nly, nzp, 3), zax=zs(2na, nly, H.nz, 3), zay=zs(nlx, 2na, H.nz, 3), mpml=FT(mpml_ratio), mdt=zs(1)) : (;))...)) : (;)
        z() = KernelAbstractions.zeros(backend, FT, nlx, nly, H.nz)
        R() = KernelAbstractions.zeros(backend, FT, NM, nlx, nly, H.nz)
        loc(A) = dev(FT.(A[xr .- ow, yr .- owy, :]))
        mem = attenuation ? (; R11=R(), R22=R(), R33=R(), R23=R(), R13=R(), R12=R()) : (;)
        vti = aniso ? merge((; wx=loc(H.wx), wz=loc(H.wz), s13=loc(H.s13), mu66=loc(H.mu66)), tti ? (; tilt=loc(H.θ), azim=loc(H.φ)) : (;)) : (;)
        r0 = H.kmap[1, 1, 2] - 2                                    # (the mesh row of node k: r0 + k)
        sponge = absorber === :cerjan ? (; gs=loc([cerjan_factor(i + ow, nx, na) * cerjan_factor(j + owy, ny, na) * cerjan_factor(r0 + k, typemax(Int), na)
                                                   for i in 1:nwx, j in 1:nwy, k in 1:H.nz])) : (;)
        merge((; V1=z(), V2=z(), V3=z(), S11=z(), S22=z(), S33=z(), S23=z(), S13=z(), S12=z(), G1=z(), G2=z(), G3=z(), J=loc(H.J),
               Jx=loc(H.Jx), Jy=loc(H.Jy), iρJ=loc(1 ./ (H.ρ .* H.J)), lam=loc(H.lam), mu=loc(aniso ? H.mu55 : H.mu),
               tp=attenuation ? loc(H.taup) : nothing, ts=attenuation ? loc(H.taus) : nothing, a1=loc(H.a1), a2=loc(H.a2),
               a3=loc(H.a3), hξ=FT(hξ), hψ=FT(hψ), hη=FT(H.hη), ox, oy, nxg=nx, nyg=ny, fluid=H.fluid), mem, vti, sponge, player)
    end)
    lite = map(B -> merge(B, (; pml=nothing)), blocks)        # (kernels without the PML: Metal allows 4 kB of arguments)
    # (kernels that see all blocks: only the arrays they read)
    recB, forceB, momB, updB, velB = slim(blocks, CV_RECORD), slim(blocks, CV_FORCE), slim(blocks, CV_STRESS), slim(blocks, CV_UPDATE), slim(blocks, CV_VEL)
    tfB = slim(blocks, CV_TF)
    below_(b) = b > 1 ? blocks[b-1] : nothing
    above_(b) = b < nb ? blocks[b+1] : nothing
    # up to 27 regions per block: low / interior / high in x, y and η, with their kernel flags (in x and y: the global
    # segments intersected with the nodes of this process, in local indices)
    m = SBP_NB
    # (PML: the layers of width p (> m) are the outer segments, with the general kernels; `bottom`: no layer at the top)
    function seg(n, a0, b0, o, p=0, bottom=false)
        S = Tuple{Int,Int,Bool,Bool}[]
        w = max(m, p)
        for (lo, hi, f, q) in ((1, w, false, p > 0), (w + 1, n - (bottom ? m : w), true, false), (n - (bottom ? m : w) + 1, n, false, p > 0 && !bottom))
            a, b = max(lo, a0), min(hi, b0)
            a <= b && push!(S, (a - o - 1, b - a + 1, f, q))
        end
        S
    end
    regions = map(enumerate(hostblocks)) do (bi, H)
        p, pz = pml ? na : 0, pml && bi == 1 ? na : 0
        # (M-PML: all three directions in all layers)
        fl(a, b, c) = (q = a[4] || b[4] || c[4]; mp ? (a[3], b[3], c[3], q, q, q, (a[4], b[4], c[4])) : (a[3], b[3], c[3], a[4], b[4], c[4]))
        Tuple(((a[1], b[1], c[1]), (a[2], b[2], c[2]), a[4] || b[4] || c[4] ? Val(fl(a, b, c)) : Val((a[3], b[3], c[3])))
              for a in seg(nx, ia, ib, ox, p), b in seg(ny, ja, jb, oy, p), c in seg(H.nz, 1, H.nz, 0, pz, true))
    end
    # (on `blks`: the blocks, or blocks of other arrays with the same mesh: the operators of the adjoint)
    function momentum!(dtT, damp, blks=blocks)
        for b in 1:nb                                  # the tractions of all blocks first (interfaces read the neighbours')
            foreach_cell!(backend, traction3_cv!, (blks[b],), (0, 0, 0), (nlx, nly, hostblocks[b].nz))
        end
        for b in 1:nb
            foreach_region!(backend, momentum3_cv!, (blks[b], neighbour3(b > 1 ? blks[b-1] : nothing), neighbour3(b < nb ? blks[b+1] : nothing), dtT, damp), regions[b])
        end
    end
    stress!(vc, dtT, blks=blocks) = for b in 1:nb
        foreach_region!(backend, stress3_cv!, (blks[b], neighbour3(b > 1 ? blks[b-1] : nothing), neighbour3(b < nb ? blks[b+1] : nothing), vc, dtT), regions[b])
    end
    fields(B) = (B.V1, B.V2, B.V3, B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)
    exchange_v!() = foreach(B -> cv_exchange!(par, (B.V1, B.V2, B.V3)), blocks)
    exchange_s!() = foreach(B -> cv_exchange!(par, (B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)), blocks)
    owned(A) = view(A, hx+1:hx+nox, hx+1:hx+noy, :)
    vnorm2() = cv_sum(par, value(sum(B -> sum(abs2, owned(B.V1)) + sum(abs2, owned(B.V2)) + sum(abs2, owned(B.V3)), blocks)))
    if dt === nothing
        rng = Random.Xoshiro(20261126 + me)                 # (seeded: the same time step in every run)
        foreach(B -> foreach(A -> copyto!(A, FT.(randn(rng, size(A)))), (B.V1, B.V2, B.V3)), blocks)
        exchange_v!()
        λ = 0.0
        for _ in 1:100
            foreach(B -> foreach(A -> fill!(A, 0), (B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)), blocks)
            stress!(nothing, one(FT))
            exchange_s!()
            nrm0 = sqrt(vnorm2())
            foreach(B -> foreach(A -> fill!(A, 0), (B.V1, B.V2, B.V3)), blocks)
            momentum!(one(FT), false)
            KernelAbstractions.synchronize(backend)
            exchange_v!()
            nrm = sqrt(vnorm2())
            λ = nrm / nrm0
            foreach(B -> foreach(A -> (A ./= FT(nrm)), (B.V1, B.V2, B.V3)), blocks)
        end
        foreach(B -> foreach(A -> fill!(A, 0), fields(B)), blocks)
        dt = 0.9 * 2 / sqrt(λ)
    end
    nt = something(nt, tmax === nothing ? nothing : round(Int, tmax / dt))
    nt === nothing && error("give the duration tmax (or nt, with dt)")
    dtT = FT(dt)
    vc = attenuation ? memory_coefficients(ts, dt, FT) : nothing
    if pml                      # the profiles of the time step (as OpenSWPC: 6 km/s, fcut the highest source frequency)
        fcut = maximum(s -> 1 / s.trise, allsrcs)
        # (M-PML: the profiles (d, α, 1/κ, d), also of the rows above the bottom layer: no damping, α of beyond it)
        pr(x, h, L) = mp ? (c = cfs_profile(x, h, L, na, fcut); FT.((c[1], c[2], 1 / c[3], c[1]))) : FT.(damping_profile(x, h, L, na, dt, fcut))
        prof(g) = (xs = (g .- g[1]) ./ 1e3; [pr(x, xs[na+1], xs[end]) for x in xs])
        copyto!.(getfield.(getfield.(blocks, :pml), :gx), Ref(prof(gx)))
        copyto!.(getfield.(getfield.(blocks, :pml), :gy), Ref(prof(gy)))
        H1 = hostblocks[1]
        # (the heights of the rows of the first column: the same on every process, exactly)
        zrow = cv_sum(par, ia == 1 && ja == 1 ? (H1.Z[1, 1, :] .- H1.Z[1, 1, 1]) ./ 1e3 : zeros(H1.nz))
        copyto!(blocks[1].pml.gz, [pr(z, zrow[na+1], 1e9) for z in zrow])
        mp && foreach(b -> fill!(blocks[b].pml.gz, pr(1e8, 1.0, 1e9)), 2:nb)
        mp && foreach(B -> fill!(B.pml.mdt, dtT), blocks)
    end
    # locate points: block and η̂ (Lagrange interpolation of the columns, bisection)
    function locate(x, y, z)
        i0, wx = lagrange_weights(gx, x)
        j0, wy = lagrange_weights(gy, y)
        (wa <= i0 && i0 + 5 <= wb && va <= j0 && j0 + 5 <= vb) ||
            error("the point ($(x / 1e3), $(y / 1e3), $(z / 1e3)) km is outside the columns of process $me")
        for (b, H) in enumerate(hostblocks)
            col = [sum(wx[a] * wy[c] * H.Z[i0+a-1-ow, j0+c-1-owy, k] for a in 1:6, c in 1:6) for k in 1:H.nz]
            (col[1] - 1e-9 <= z <= col[end] + 1e-9) || continue
            gη, _ = sbp_grid(0.0, 1.0, H.nz)
            zof(η) = let (q0, wq) = lagrange_weights(gη, η)
                sum(wq[c] * col[q0+c-1] for c in 1:6)
            end
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                zof(mid) < z ? (lo = mid) : (hi = mid)
            end
            return b, (lo + hi) / 2
        end
        error("the point ($(x / 1e3), $(y / 1e3), $(z / 1e3)) km is outside the model")
    end
    function source_table(list, coef, δ=(0.0, 0.0, 0.0))
        tab = (b=Int[], i=Int[], j=Int[], k=Int[], c=Float64[], n=Int[])
        for (n, s) in enumerate(list)
            x, y, z = s.x + δ[1], s.y + δ[2], s.z + δ[3]
            iξ, wξ = delta_weights(gx, Hξ, hξ, 1e3x)
            iψ, wψ = delta_weights(gy, Hψ, hψ, 1e3y)
            (any(i -> ia <= i <= ib, iξ) && any(j -> ja <= j <= jb, iψ)) || continue     # (no node of this process)
            b, η = locate(1e3x, 1e3y, 1e3z)
            H = hostblocks[b]
            gη, hη = sbp_grid(0.0, 1.0, H.nz)
            iη, wη = delta_weights(gη, sbp_weights(H.nz, hη), hη, η)
            for (a, i) in enumerate(iξ), (c, j) in enumerate(iψ), (d, k) in enumerate(iη)
                push!(tab.b, b); push!(tab.i, i); push!(tab.j, j); push!(tab.k, k); push!(tab.n, n)
                push!(tab.c, wξ[a] * wψ[c] * wη[d] * coef(H, i, j, k))
            end
        end
        return tab
    end
    # teleseismic wave: total-field / scattered-field corrections at the nodes near the box boundary
    column(ig, jg) = (cz = model_columns(model, ig:ig, jg:jg, 3); (vec(1e3 .* cz[1]),
                      [(Float64(cz[2].rho[1, 1, q]) * (density_unit === :g_cm3 ? 1e3 : 1.0), 1e3cz[2].vp[1, 1, q], 1e3cz[2].vs[1, 1, q]) for q in axes(cz[1], 3)]))
    TW, ntw = isempty(tws) ? (nothing, 0) :
              teleseismic_cv(tws[1], 3, false, (; hostblocks, gx, nx, ia, ib, ow, ox, gy, ny, ja, jb, owy, oy, dev, FT, dt, nt, tbeg, stf, aniso,
                                                 blocks, na, pml, column, rows, loc=A -> A[xr .- ow, yr .- owy, :]))[1:2]
    forces = filter(s -> s isa ForceSource, srcs)
    moments = filter(s -> s isa MomentTensorSource, srcs)
    mine(t) = (q = findall(n -> ia <= t.i[n] <= ib && ja <= t.j[n] <= jb, eachindex(t.i));
               by_node((b=t.b[q], i=t.i[q] .- ox, j=t.j[q] .- oy, k=t.k[q], c=t.c[q], n=t.n[q])))
    fcoef, mcoef = (H, i, j, k) -> 1.0, (H, i, j, k) -> 1 / H.J[i-ow, j-owy, k]
    tables(δ) = (mine(source_table(forces, fcoef, δ)), mine(source_table(moments, mcoef, δ)))
    ft, mt = tables((0.0, 0.0, 0.0))
    # the amplitudes per source and step, on the device in windows of steps (source_window.jl)
    amp_f(n) = (s = forces[n]; amp_row(source_stf(s, stf), 1, 1, 1, tbeg, 1, dt, s.t0, s.trise))
    amp_m(n) = (s = moments[n]; amp_row(source_stf(s, stf), 1, dt, 1, tbeg, 0.5, dt, s.t0, s.trise))
    Wf, Wm = source_window(amp_f, length(forces), nt, dev, FT, backend), source_window(amp_m, length(moments), nt, dev, FT, backend)
    FS = (; b=dev(ft.b), i=dev(ft.i), j=dev(ft.j), k=dev(ft.k), c=dev(FT.(ft.c)), n=dev(ft.n), seg=dev(ft.seg), F=dev(FT[s.F[c] for s in forces, c in 1:3]),
          s=Wf.s)
    MS = (; b=dev(mt.b), i=dev(mt.i), j=dev(mt.j), k=dev(mt.k), c=dev(FT.(mt.c)), n=dev(mt.n), seg=dev(mt.seg), M=dev(FT[s.M0 * s.M[c] for s in moments, c in 1:6]),
          s=Wm.s)
    nst = length(stations)
    st = map(stations) do s
        i0, wx = lagrange_weights(gx, 1e3s.x)
        j0, wy = lagrange_weights(gy, 1e3s.y)
        near = i0 + 5 >= ia && i0 <= ib && j0 + 5 >= ja && j0 <= jb          # (other stations: not recorded on this process)
        b, η = s.surface || !near ? (nb, 1.0) : locate(1e3s.x, 1e3s.y, 1e3s.z)
        gη, _ = sbp_grid(0.0, 1.0, hostblocks[b].nz)
        k0, wz = lagrange_weights(gη, η)
        (b, i0, j0, k0, wx, wy, wz)
    end
    st_all = st
    own = findall(s -> ia <= s[2] + 2 <= ib && ja <= s[3] + 2 <= jb, st)     # the stations of this process
    st = st[own]
    nst = length(st)
    St = (; b=dev([s[1] for s in st]), i0=dev([s[2] - ox for s in st]), j0=dev([s[3] - oy for s in st]), k0=dev([s[4] for s in st]),
          wx=dev(FT[s[5][a] for a in 1:6, s in st]), wy=dev(FT[s[6][a] for a in 1:6, s in st]), wz=dev(FT[s[7][a] for a in 1:6, s in st]))
    nrec = cld(nt, record_every)
    rec = KernelAbstractions.zeros(backend, FT, nrec, 18, nst)
    acc = KernelAbstractions.zeros(backend, FT, 3, nst)
    prev = KernelAbstractions.zeros(backend, FT, 3, nst)
    function velocity_field()                       # (with MPI: gathered on rank 0, `nothing` elsewhere)
        keep(b) = b == 1 ? (1:hostblocks[b].nz) : (2:hostblocks[b].nz)
        r(c) = Float64.(cat((Array(owned(getfield(blocks[b], c)))[:, :, keep(b)] for b in 1:nb)...; dims=3))
        V = (r(:V1), r(:V2), r(:V3))
        zz = Zw[ia-ow:ib-ow, ja-owy:jb-owy, :]
        if par !== nothing
            parts = MPI.gather((ia, ib, ja, jb, V, zz), par.comm; root=0)
            me == 0 || return nothing
            V, zz = ntuple(_ -> zeros(nx, ny, nzt), 3), zeros(nx, ny, nzt)
            for (a, b, c, d, p, pz) in parts
                foreach(q -> V[q][a:b, c:d, :] .= p[q], 1:3)
                zz[a:b, c:d, :] .= pz
            end
        end
        Xg, Yg = [gx[i] / 1e3 for i in 1:nx, _ in 1:ny, _ in 1:nzt], [gy[j] / 1e3 for _ in 1:nx, j in 1:ny, _ in 1:nzt]
        return CartData(Xg, Yg, zz, (; V))
    end
    planes = snapshots === nothing ? nothing :
             setup_planes_cv(snapshots, 3, (; hostblocks, nx, ny=ny, gx, gy=gy, Zm, ia, ib, ja=ja, jb, oy, ox, dev, backend, FT, dir=snapshot_dir,
                                              me, parallel=par !== nothing, par, ow, owy))
    Dps, divrot! = planes !== nothing && :ps in snapshots.quantities ? divrot_cv(blocks, stress!, 3, backend, FT) : (nothing, nothing)
    movie = (snapshot_every > 0 && callback === nothing && snapshots === nothing && me == 0) ?
            movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    verbose &= me == 0
    verbose && @printf("SeismicWavePC: 3D on a curvilinear grid, %d × %d × %d nodes in %d block(s) (interfaces at rows %s), %d steps of %.4g s%s%s\n",
                       nx, ny, nzt, nb, isempty(rows) ? "–" : join(rows, ", "), nt, dt, attenuation ? ", attenuation" : "",
                       absorber === :none ? "" : absorber === :cerjan ? ", absorbing sides and bottom (sponge of $na nodes)" : ", absorbing sides and bottom")
    # one step (n = it - 1); recording if `record`
    function step!(it, record)
        n = it - 1
        itw = record && n % record_every == 0 ? n ÷ record_every + 1 : 0
        itw > 0 && nst > 0 && foreach_item!(backend, vprev3_cv!, (prev, recB, St), nst)
        momentum!(dtT, true)
        ntw > 0 && foreach_item!(backend, tf_momentum3!, (tfB, TW, FT(tbeg + n * dt), dtT), ntw)
        absorber === :cerjan && foreach(B -> (B.V1 .*= B.gs; B.V2 .*= B.gs; B.V3 .*= B.gs), blocks)
        length(FS.seg) > 1 && foreach_item!(backend, force3_cv!, (forceB, FS, window_step!(Wf, it), dtT), length(FS.seg) - 1)
        par === nothing || KernelAbstractions.synchronize(backend)
        exchange_v!()
        record && nst > 0 && foreach_item!(backend, record3_cv!, (rec, acc, prev, recB, St, itw, dtT), nst)
        stress!(vc, dtT)
        ntw > 0 && foreach_item!(backend, tf_stress3!, (updB, TW, vc, FT(tbeg + (n + 0.5) * dt), dtT), ntw)
        absorber === :cerjan && foreach(B -> foreach(S -> S .*= B.gs, (B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)), blocks)
        length(MS.seg) > 1 && foreach_item!(backend, moment3_cv!, (momB, MS, window_step!(Wm, it)), length(MS.seg) - 1)
        par === nothing || KernelAbstractions.synchronize(backend)
        exchange_s!()
    end
    function all_traces(part=identity)
        traces = zeros(nrec, 18, length(stations))
        traces[:, :, own] .= part.(Array(rec))
        par === nothing || MPI.Allreduce!(traces, +, par.comm)
        return traces
    end
    seismograms(traces, wf=nothing) = Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), traces[:, 1:3, :], traces[:, 4:6, :],
                                                  traces[:, 7:12, :], traces[:, 13:18, :], collect(stations), (:x, :y, :z),
                                                  (:xx, :yy, :zz, :yz, :xz, :xy), wf)
    driver === nothing || return drive_curvilinear3(driver, (; backend, FT, blocks, hostblocks, nb, nx, ny, nzt, nlx, nly, nox, noy,
        hx, ox, oy, xr, yr, ia, ib, ja, jb, par, me, dt, dtT, nt, nrec, record_every, momentum!, stress!, step!, exchange_v!,
        exchange_s!, FS, forces, Hξ, Hψ, tbeg, attenuation, dev, model, all_traces, stations, density_unit, nsrc=length(allsrcs), st_all, sponge=absorber === :cerjan,
        seismograms, ts, fq_ref, vc, loc=A -> A[xr .- ow, yr .- owy, :], ftab=ft, mtab=mt, MS, moments, srcs=allsrcs, stf, tables, ow, owy, Wf, Wm, amp_f, amp_m,
        TW, ntw, tfkind=Val(:p3d)))
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
                isempty(p.own) || foreach_item!(backend, plane_cv!, (p.B, p.U, p.PK, velB, store ? Dps : nothing, p.pts, dtT, store, Val(3)),
                                                length(p.own))
            end
            if store
                KernelAbstractions.synchronize(backend)
                write_planes(planes, :p3d, n ÷ snapshot_every + 1, tbeg + n * dt, 1.0, par !== nothing, me)
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
            vmax = cv_max(par, Float64(maximum(B -> max(maximum(abs, owned(B.V1)), maximum(abs, owned(B.V2)), maximum(abs, owned(B.V3))), blocks)))
            isfinite(vmax) || error("curvilinear grid: the simulation became unstable at step $it")
            verbose && @printf("  step %6d/%d  t = %8.4f s  max|v| = %.3e m/s\n", it, nt, tbeg + it * dt, vmax)
        end
    end
    KernelAbstractions.synchronize(backend)
    movie === nothing || movie_paraview(movie)
    planes === nothing || write_peaks(planes, snapshots.quantities, :p3d, 1.0, par !== nothing, me)
    verbose && @printf("  finished in %.2f s\n", time() - wall)
    return seismograms(all_traces(), wavefield ? velocity_field() : nothing)
end
