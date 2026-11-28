# Misfit gradients on curvilinear grids (2D P-SV and 3D): the exact discrete adjoint of the leapfrog scheme of
# curvilinear.jl and curvilinear3d.jl. The scheme conserves energy, E = ½ Σ H (ρJ |v|² + J σ:C⁻¹σ), so its momentum operator A (divergence,
# all SAT terms: ρJ ∂t v = A σ) and gradient operator G (strains, with the interface penalty: ∂t σ = C G v) satisfy
#     H A = -(H J G)ᵀ   (in the pairing σ11 exx + σ22 ezz + σ12 exz),
# i.e. Aᵀ = -(H J) G H⁻¹ and Gᵀ = -H A (H J)⁻¹: the adjoint applies the forward kernels (on "operator blocks" with
# ρJ = 1 for A, and λ = 0, μ = ½ for G) and pointwise scalings. The implicit impedance damping is diagonal. The forward
# wavefield comes from checkpoints, recomputed segment by segment. Attenuation: adjoint memory variables (the transposed
# Crank-Nicolson recursion; pointwise). Gradients with respect to λ, μ, τp, τs, 1/(ρJ) and the damping (vp, vs at the
# boundary nodes) are turned into ρ, vp, vs, Qp, Qs on the nodes of the model.

struct CurvilinearGradient{O,M}
    observed::O
    misfit::M
    checkpoint_steps::Union{Int,Nothing}
    illumination::Bool
end

"""
Add the residual (∂J/∂v at the recorded samples) × `s` to the adjoint velocities at the stations (transposed interpolation).
(MPI: the weights of the nodes of other processes are zero; a stencil may reach beyond the halo there: skipped.)
"""
Base.@propagate_inbounds function residual_cv!(blocks, St, R, itw, s)
    for n in 1:length(St.b)
        B = blocks[St.b[n]]
        for c in 1:6, a in 1:6
            iszero(St.wx[a, n]) && continue
            w = s * St.wx[a, n] * St.wz[c, n]
            i, k = St.i0[n] + a - 1, St.k0[n] + c - 1
            B.V1[i, k] += w * R[itw, 1, n]
            B.V2[i, k] += w * R[itw, 2, n]
        end
    end
end
"The forces depend on 1/(ρJ) at their nodes: their part of the gradient."
Base.@propagate_inbounds function force_grad_cv!(G, A, S, it, dt)
    for m in 1:length(S.i)
        b, i, k, n = S.b[m], S.i[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it]
        G[b].iρJ[i, k] += s * (S.F[n, 1] * A[b].V1[i, k] + S.F[n, 2] * A[b].V2[i, k])
    end
end

Base.@propagate_inbounds function residual3_cv!(blocks, St, R, itw, s)
    for n in 1:length(St.b)
        B = blocks[St.b[n]]
        for c in 1:6, b in 1:6, a in 1:6
            (iszero(St.wx[a, n]) || iszero(St.wy[b, n])) && continue
            w = s * St.wx[a, n] * St.wy[b, n] * St.wz[c, n]
            i, j, k = St.i0[n] + a - 1, St.j0[n] + b - 1, St.k0[n] + c - 1
            B.V1[i, j, k] += w * R[itw, 1, n]
            B.V2[i, j, k] += w * R[itw, 2, n]
            B.V3[i, j, k] += w * R[itw, 3, n]
        end
    end
end
Base.@propagate_inbounds function force3_grad_cv!(G, A, S, it, dt)
    for m in 1:length(S.i)
        b, i, j, k, n = S.b[m], S.i[m], S.j[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it]
        G[b].iρJ[i, j, k] += s * (S.F[n, 1] * A[b].V1[i, j, k] + S.F[n, 2] * A[b].V2[i, j, k] + S.F[n, 3] * A[b].V3[i, j, k])
    end
end

"SH: the residual on vy, and the force gradient (fy)."
Base.@propagate_inbounds function residual_sh_cv!(blocks, St, R, itw, s)
    for n in 1:length(St.b)
        B = blocks[St.b[n]]
        for c in 1:6, a in 1:6
            iszero(St.wx[a, n]) && continue
            B.V1[St.i0[n]+a-1, St.k0[n]+c-1] += s * St.wx[a, n] * St.wz[c, n] * R[itw, 1, n]
        end
    end
end
Base.@propagate_inbounds function force_grad_sh_cv!(G, A, S, it, dt)
    for m in 1:length(S.i)
        b, i, k, n = S.b[m], S.i[m], S.k[m], S.n[m]
        G[b].iρJ[i, k] += dt * S.c[m] * S.s[n, it] * S.F[n, 1] * A[b].V1[i, k]
    end
end

# Source sensitivities (per entry m of the source tables: the time integrals of the adjoint field weighted by the source
# time function, per component; per source and step: ∂J/∂(moment rate))
@inline node_of(S, m) = haskey(S, :j) ? CartesianIndex(S.i[m], S.j[m], S.k[m]) : CartesianIndex(S.i[m], S.k[m])
Base.@propagate_inbounds function force_sens_cv!(AF, gm, Adj, blocks, S, it, dt, ::Val{vn}) where {vn}
    for m in 1:length(S.i)
        b, n, I = S.b[m], S.n[m], node_of(S, m)
        w = dt * blocks[b].iρJ[I]
        v = map(k -> getfield(Adj[b], k)[I], vn)
        for c in eachindex(vn)
            AF[m, c] += w * S.s[n, it] * v[c]
            gm[n, it] += w * S.c[m] * S.F[n, c] * v[c]
        end
    end
end
Base.@propagate_inbounds function moment_sens_cv!(AM, gm, Adj, S, it, dt, ::Val{sn}) where {sn}
    for m in 1:length(S.i)
        b, n, I = S.b[m], S.n[m], node_of(S, m)
        σ = map(k -> getfield(Adj[b], k)[I], sn)
        for c in eachindex(sn)
            AM[m, c] += S.s[n, it] * σ[c]
            gm[n, it] -= dt * S.c[m] * S.M[n, c] * σ[c]
        end
    end
end

drive_curvilinear(d::CurvilinearGradient, c) = curvilinear_gradient(d, c, 2)
drive_curvilinear3(d::CurvilinearGradient, c) = curvilinear_gradient(d, c, 3)

# TTI (anisotropy.jl): the stress update is σ += M F(Mᵀ e) dt with F the VTI update in the frame of the axis, M the rotation
# of the stress back (σ = Aᵀ σ' A) and Mᵀ that of engineering strains. Its transpose rotates the adjoint stress P and the
# sums Q of the adjoint memory variables (both pair with the stress: engineering, as strains) and the strains e into the
# frame of the axis, applies the VTI adjoint there and rotates its output Y (a stress) back. The gradient with respect to
# an angle: dt [(Ė e)·(Cp Zp' + Cs X') + e'·(Cp Ė Zp + Cs Ė X)], Ė the derivative of Mᵀ, Zp, X the adjoint stresses that
# weigh the parts of C relaxing with τp and τs (as in the VTI adjoint). `pos[k]`: the stress array of the 3D component
# k = (xx, yy, zz, yz, xz, xy) (0: none; 2D P-SV: xx, zz, xz; SH: yz, xy), the axis tilts in x-z in 2D (φ = 0).
@inline emb6(X, n, ::Val{pos}, z) where {pos} = ntuple(k -> pos[k] == 0 ? z : X[pos[k]][n], Val(6))
@inline function set6!(X, n, v, ::Val{pos}) where {pos}
    ntuple(Val(6)) do k
        pos[k] == 0 || (X[pos[k]][n] = v[k])
        nothing
    end
    return nothing
end
@inline tti_angles(T, n) = (T.tilt[n], T.azim === nothing ? zero(eltype(T.tilt)) : T.azim[n])
Base.@propagate_inbounds function tti_adjoint_in!(Pl, Ql, el, P, Q, e, T, vp, n)
    A = axis_frame(tti_angles(T, n)...)
    z = zero(eltype(T.tilt))
    set6!(Pl, n, full_shear(rot3(A, half_shear(emb6(P, n, vp, z)))), vp)
    set6!(Ql, n, full_shear(rot3(A, half_shear(emb6(Q, n, vp, z)))), vp)
    set6!(el, n, rot3(A, emb6(e, n, vp, z)), vp)
end
Base.@propagate_inbounds function tti_adjoint_out!(Y, gθ, gφ, Pl, Ql, el, P, Q, e, M, τp, τs, d2, dt, T, vp, n)
    θ, φ = tti_angles(T, n)
    A = axis_frame(θ, φ)
    z = zero(θ)
    set6!(Y, n, rot3(transpose3(A), emb6(Y, n, vp, z)), vp)
    tp, ts = τp === nothing ? z : τp[n], τs === nothing ? z : τs[n]
    wx, wz, s13, mu, m66 = map(X -> X[n], M)
    Pg, Qg, Pq, Qq = emb6(P, n, vp, z), emb6(Q, n, vp, z), emb6(Pl, n, vp, z), emb6(Ql, n, vp, z)
    wp(P_, Q_) = map((p, q) -> (1 + tp * (1 + d2)) * p - tp * q, P_, Q_)
    ws(P_, Q_) = map((p, q) -> (1 + ts * (1 + d2)) * p - ts * q, P_, Q_)
    Zg, Xg, Zl, Xl = wp(Pg, Qg), ws(Pg, Qg), wp(Pq, Qq), ws(Pq, Qq)
    eg, elq = emb6(e, n, vp, z), full_shear(emb6(el, n, vp, z))      # (e: tensor; el engineering)
    CZ = map(+, vti_split(wx, wz, s13, mu, m66, Zl)[1], vti_split(wx, wz, s13, mu, m66, Xl)[2])
    dθ, dφ = axis_frame_derivatives(θ, φ)
    g(Ad) = (Ee = full_shear(drot3(A, Ad, eg)); EZ = full_shear(drot3(A, Ad, half_shear(Zg))); EX = full_shear(drot3(A, Ad, half_shear(Xg)));
             dt * (dot6(Ee, CZ) + dot6(elq, map(+, vti_split(wx, wz, s13, mu, m66, EZ)[1], vti_split(wx, wz, s13, mu, m66, EX)[2]))))
    gθ[n] += g(dθ)
    gφ === nothing || (gφ[n] += g(dφ))
end

function curvilinear_gradient(d::CurvilinearGradient, c, N)
    (; backend, FT, blocks, hostblocks, nb, nx, nzt, hx, ox, par, dtT, nt, record_every, momentum!, stress!, step!, FS, forces,
       Hξ, attenuation, dev, model, all_traces, density_unit, vc, ts, fq_ref, loc) = c
    ny, oy = N == 2 ? (1, 0) : (c.ny, c.oy)
    vn = N == 2 ? (:V1, :V2) : (:V1, :V2, :V3)
    sn = N == 2 ? (:S11, :S22, :S12) : (:S11, :S22, :S33, :S23, :S13, :S12)       # normal stresses first
    rn = N == 2 ? (:R11, :R22, :R12) : (:R11, :R22, :R33, :R23, :R13, :R12)
    an = N == 2 ? (:a1, :a2) : (:a1, :a2, :a3)
    sh = haskey(blocks[1], :S1)                                      # SH: vy, σxy, σyz (2D)
    sh && ((vn, sn, rn, an) = ((:V1,), (:S1, :S2), (:R1, :R2), (:a1,)))
    nV, nN = length(vn), sh ? 0 : N                                   # velocity components, normal stresses
    nS = length(sn)
    comps(B, names) = map(k -> getfield(B, k), names)
    zero_(b, n...) = KernelAbstractions.zeros(backend, FT, n..., size(blocks[b].J)...)
    fill_(b, v) = (A = zero_(b); fill!(A, FT(v)); A)
    fields(B) = (comps(B, vn)..., comps(B, sn)..., (attenuation ? comps(B, rn) : ())..., pml_arrays(B)...)
    # 1. forward, with checkpoints (host) every K steps; the misfit and its residual
    K = something(d.checkpoint_steps, max(1, ceil(Int, sqrt(nt))))
    ckpt = Dict{Int,Any}()
    for it in 1:nt
        (it - 1) % K == 0 && (ckpt[it] = [Array.(fields(B)) for B in blocks])
        step!(it, true)
    end
    J, adj = d.misfit(all_traces()[:, 1:nV, :], d.observed.v, d.observed.t)
    # the residuals go to the nodes of this process of every station whose stencil reaches them (a station near a
    # subdomain boundary feeds both sides; the forward run reads the neighbour's nodes from the halo instead)
    sa = c.st_all
    inx(s) = s[2] <= c.ib && s[2] + 5 >= c.ia
    wx(s, a) = (c.ia <= s[2] + a - 1 <= c.ib) * s[N+2][a]
    if N == 2
        sel = [n for (n, s) in enumerate(sa) if inx(s)]
        St = (; b=dev([sa[n][1] for n in sel]), i0=dev([sa[n][2] - ox for n in sel]), k0=dev([sa[n][3] for n in sel]),
              wx=dev(FT[wx(sa[n], a) for a in 1:6, n in sel]), wz=dev(FT[sa[n][5][a] for a in 1:6, n in sel]))
        residual!, force_grad! = sh ? (residual_sh_cv!, force_grad_sh_cv!) : (residual_cv!, force_grad_cv!)
    else
        sel = [n for (n, s) in enumerate(sa) if inx(s) && s[3] <= c.jb && s[3] + 5 >= c.ja]
        St = (; b=dev([sa[n][1] for n in sel]), i0=dev([sa[n][2] - ox for n in sel]), j0=dev([sa[n][3] - oy for n in sel]),
              k0=dev([sa[n][4] for n in sel]), wx=dev(FT[wx(sa[n], a) for a in 1:6, n in sel]),
              wy=dev(FT[(c.ja <= sa[n][3] + a - 1 <= c.jb) * sa[n][6][a] for a in 1:6, n in sel]),
              wz=dev(FT[sa[n][7][a] for a in 1:6, n in sel]))
        residual!, force_grad! = residual3_cv!, force3_grad_cv!
    end
    nst = length(sel)
    Rres = dev(FT.(adj[:, :, sel]))
    # 2. the adjoint fields, operator blocks and gradient accumulators
    # (host arrays: the columns of the window of this process, index - (ow, owy))
    weights(H) = N == 2 ? [Hξ[i+c.ow] * h for i in axes(H.kmap, 1), h in sbp_weights(H.nz, H.hη)] :
                          [Hξ[i+c.ow] * c.Hψ[j+c.owy] * h for i in axes(H.kmap, 1), j in axes(H.kmap, 2), h in sbp_weights(H.nz, H.hη)]
    Hw = [dev(FT.(loc(weights(H)))) for H in hostblocks]                                    # SBP norm per node
    HJ = [Hw[b] .* blocks[b].J for b in 1:nb]
    Adj = Tuple(NamedTuple{(vn..., sn...)}(ntuple(_ -> zero_(b), nV + nS)) for b in 1:nb)
    Rb = Tuple(attenuation ? ntuple(_ -> zero_(b, NM), nS) : () for b in 1:nb)            # adjoint memory variables
    Qm = Tuple(ntuple(_ -> zero_(b), nS) for b in 1:nb)                                      # Σₘ c2ₘ R̄ₘ (0: elastic)
    Wk = Tuple(ntuple(_ -> zero_(b), 4) for b in 1:nb)                                       # work arrays
    Xw, Xpw = (Tuple(ntuple(_ -> zero_(b), nS) for b in 1:nb) for _ in 1:2)
    # the moduli in the VTI form (isotropic media: wx = wz = √(λ+2μ), s13 = -2μ, mu66 = μ): one adjoint for both
    Mv = Tuple(map(A -> dev(FT.(loc(A))), (H.wx, H.wz, H.s13, H.mu55, H.mu66)) for H in hostblocks)
    tilted_ = haskey(blocks[1], :tilt)                               # TTI: rotate into the frame of the axis
    tilt3 = tilted_ && N == 3
    Grad = Tuple(merge((; wx=zero_(b), wz=zero_(b), s13=zero_(b), mu=zero_(b), m66=zero_(b), tp=zero_(b), ts=zero_(b), iρJ=zero_(b),
                          a=ntuple(_ -> zero_(b), nV)), tilted_ ? (; tilt=zero_(b), azim=tilt3 ? zero_(b) : nothing) : (;)) for b in 1:nb)
    # (TTI: P, Q and the strains in the frame of the axis; the stress arrays of the 3D components)
    Loc = tilted_ ? Tuple(ntuple(_ -> ntuple(_ -> zero_(b), nS), 3) for b in 1:nb) : nothing
    vpos = Val(sh ? (0, 0, 0, 2, 0, 1) : N == 2 ? (1, 0, 2, 0, 3, 0) : (1, 2, 3, 4, 5, 6))
    angles(b) = (; tilt=blocks[b].tilt, azim=tilt3 ? blocks[b].azim : nothing)
    Win = Tuple(ntuple(_ -> zero_(b), nV) for b in 1:nb)        # inputs of G (velocities)
    Gout = Tuple(ntuple(_ -> zero_(b), nS) for b in 1:nb)      # its strains (normal; shear halved)
    Yin = Tuple(ntuple(_ -> zero_(b), nS) for b in 1:nb)       # inputs of A (stresses)
    Aout = Tuple(ntuple(_ -> zero_(b), nV) for b in 1:nb)       # A y
    nomem = attenuation ? NamedTuple{rn}(ntuple(_ -> nothing, nS)) : (;)
    Gblk = Tuple(merge(Base.structdiff(blocks[b], NamedTuple{(:wx, :wz, :s13, :mu66, :tilt, :azim)}), NamedTuple{vn}(Win[b]), NamedTuple{sn}(Gout[b]), nomem,
                       (; lam=zero_(b), mu=fill_(b, 0.5), mu66=fill_(b, 0.5), tp=nothing, ts=nothing, pml=nothing)) for b in 1:nb)
    tractions(b) = N == 3 ? (; G1=zero_(b), G2=zero_(b), G3=zero_(b)) : (;)                 # (3D: the η tractions of A)
    Ablk = Tuple(merge(blocks[b], NamedTuple{sn}(Yin[b]), NamedTuple{vn}(Aout[b]), (; iρJ=fill_(b, 1), pml=nothing), tractions(b)) for b in 1:nb)
    # PML: the forward memory variables (copies: `scratch`) make the operators give the stretched strains and divergences of
    # the forward; the adjoint applies the parts of the operators of the stretched directions separately (`part`)
    pml = pmlof(blocks[1]) !== nothing
    lite = map(B -> merge(B, (; pml=nothing)), blocks)
    memnames(P) = Tuple(k for k in keys(P) if getfield(P, k) isa AbstractArray && !(k in (:gx, :gy, :gz, :mdt)))
    scratch = pml ? Tuple(merge(blocks[b].pml, NamedTuple{memnames(blocks[b].pml)}(map(copy, pml_arrays(blocks[b])))) for b in 1:nb) : nothing
    withpml(blk, b) = pml ? merge(blk, (; pml=scratch[b])) : blk
    AblkF = Tuple(withpml(merge(Ablk[b], NamedTuple{sn}(comps(blocks[b], sn))), b) for b in 1:nb)       # A σ of the forward
    GblkP = Tuple(withpml(Gblk[b], b) for b in 1:nb)                                                 # its strains
    "G w: the strains of the velocities in Win (exchanged first); shear strains in Gout halved"
    function Gop!(blk=Gblk)
        foreach(b -> cv_exchange!(par, Win[b]), 1:nb)
        foreach(g -> foreach(A -> fill!(A, 0), g), Gout)
        stress!(nothing, one(FT), blk)
    end
    function Aop!(blk)
        blk === AblkF || foreach(b -> cv_exchange!(par, Yin[b]), 1:nb)
        foreach(a -> foreach(A -> fill!(A, 0), a), Aout)
        momentum!(one(FT), false, blk)
    end
    # the stretched directions: (part, the pieces of every block: local ranges, γ₁ - 1, γ₂, γ₃, γ₄ dt on them) and the
    # adjoint memory variables of the momentum (per velocity) and the stress (per stress component) of every piece. A piece:
    # the nodes of a layer where the forward kernels stretch that direction (PML: ξ in the side layers, η in the bottom one;
    # M-PML: every direction in all layers), with the coefficients of the forward kernels (`pml_xgamma`, ...).
    dirs = pml ? (N == 2 ? (:xi, :eta) : (:xi, :psi, :eta)) : ()
    function pieces(D, b)
        P = blocks[b].pml
        na, mp = P.np, ismpml(P)
        nz = hostblocks[b].nz
        gxh, gzh = Array(P.gx), Array(P.gz)
        gyh = N == 3 ? Array(P.gy) : nothing
        xo = c.ia:c.ib
        yo = N == 3 ? (c.ja:c.jb) : (1:1)
        ny_ = N == 3 ? c.ny : 1
        cut(r, o) = max(first(r), first(o)):min(last(r), last(o))
        xs = filter(!isempty, [cut(1:na, xo), cut(nx-na+1:nx, xo)]); xm = cut(na+1:nx-na, xo)
        ys = N == 3 ? filter(!isempty, [cut(1:na, yo), cut(ny_-na+1:ny_, yo)]) : UnitRange{Int}[]
        ym = N == 3 ? cut(na+1:ny_-na, yo) : yo
        all_, bot, above = 1:nz, b == 1 ? (1:na) : (1:0), b == 1 ? (na+1:nz) : (1:nz)
        sets = if N == 2
            D === :xi ? [[(x, all_) for x in xs]; mp ? [(xm, bot)] : []] :
                        [[(xo, bot)]; mp ? [(x, above) for x in xs] : []]
        elseif D === :xi
            [[(x, yo, all_) for x in xs]; mp ? [[(xm, y, all_) for y in ys]; [(xm, ym, bot)]] : []]
        elseif D === :psi
            [[(xo, y, all_) for y in ys]; mp ? [[(x, ym, all_) for x in xs]; [(xm, ym, bot)]] : []]
        else
            [[(xo, yo, bot)]; mp ? [[(x, yo, above) for x in xs]; [(xm, y, above) for y in ys]] : []]
        end
        Pm = mp ? (; mpml=P.mpml) : nothing
        # (the coefficients of the forward kernels at the node (ig, jg, k), the sums of d∥ in the same order)
        γ(ig, jg, k) = if !mp
            D === :xi ? gxh[ig] : D === :psi ? gyh[jg] : gzh[k]
        elseif N == 2
            D === :xi ? mpml_gamma(Pm, gxh[ig], gzh[k][4], dtT) : mpml_gamma(Pm, gzh[k], gxh[ig][4], dtT)
        else
            D === :xi ? mpml_gamma(Pm, gxh[ig], gyh[jg][4] + gzh[k][4], dtT) :
            D === :psi ? mpml_gamma(Pm, gyh[jg], gxh[ig][4] + gzh[k][4], dtT) : mpml_gamma(Pm, gzh[k], gxh[ig][4] + gyh[jg][4], dtT)
        end
        out = []
        for st in sets
            any(isempty, st) && continue
            G = N == 2 ? [γ(i, 1, k) for i in st[1], k in st[2]] : [γ(i, j, k) for i in st[1], j in st[2], k in st[3]]
            F = N == 2 ? (st[1] .- ox, st[2]) : (st[1] .- ox, st[2] .- oy, st[3])
            push!(out, (F, dev(FT[g[1] - 1 for g in G]), dev(FT[g[2] for g in G]), dev(FT[g[3] for g in G]), dev(FT[g[4] * dtT for g in G])))
        end
        return out
    end
    PML = Tuple(begin
                    pcs = Tuple(pieces(D, b) for b in 1:nb)
                    mem(n) = Tuple(Tuple(ntuple(_ -> KernelAbstractions.zeros(backend, FT, length.(pc[1])...), n) for pc in pcs[b]) for b in 1:nb)
                    (; part=Val(D), pieces=pcs, mv=mem(nV), ms=mem(nS))
                end for D in dirs)
    Win0 = pml ? Tuple(ntuple(_ -> zero_(b), nV) for b in 1:nb) : nothing
    Yin0 = pml ? Tuple(ntuple(_ -> zero_(b), nS) for b in 1:nb) : nothing
    "W = (γ₁ - 1) W0 + γ₄ dt M̄, M̄ ← γ₂ W0 + γ₃ M̄ on the pieces of the direction (W = 0 elsewhere)"
    function pml_adjoint!(W, W0, M, pcs)
        for b in 1:nb, k in eachindex(W[b])
            fill!(W[b][k], 0)
            for (p, (F, g1, g2, g3, g4)) in enumerate(pcs[b])
                w, r, m = view(W[b][k], F...), view(W0[b][k], F...), M[b][p][k]
                @. w = g1 * r + g4 * m
                @. m = g2 * r + g3 * m
            end
        end
    end
    MS, nF, nM = c.MS, length(forces), length(c.moments)
    # ∂J/∂(moment rate) per source and step (gmF, gmM): in windows of steps of the backward pass, summed on the host
    gmF, gmM = zeros(nF, nt), zeros(nM, nt)
    collect_gm(gm) = (outs, steps, _) -> (gm[:, steps] .+= view(outs[1], :, 1:length(steps)))
    Wfb = source_window(c.amp_f, nF, nt, c.dev, FT, backend; outputs=((nF,),), reduce=collect_gm(gmF))
    Wmb = source_window(c.amp_m, nM, nt, c.dev, FT, backend; outputs=((nM,),), reduce=collect_gm(gmM))
    FSb, MSb = merge(FS, (; s=Wfb.s)), merge(MS, (; s=Wmb.s))
    AF = KernelAbstractions.zeros(backend, FT, length(c.ftab.i), nV)
    AM = KernelAbstractions.zeros(backend, FT, length(c.mtab.i), nS)
    L = d.illumination ? Tuple(zero_(b) for b in 1:nb) : nothing                              # Σₙ |v^{n+1/2}|²
    c1, c2, d1, d2 = attenuation ? vc : (nothing, nothing, nothing, zero(FT))
    # the transposed stress update σ^{n+1} = σⁿ + dt (C₁ e + Σₘ d1ₘ Rₘ^{n+1}), Rₘ^{n+1} = c1ₘ Rₘⁿ - c2ₘ dt Cτ e with the
    # strains e = G v^{n+1/2} in Gout (shear: γ/2) and σ̄ = P: adjoint memory variables, the gradient with respect to the
    # moduli (VTI form) and τp, τs, and Y = (C₁ P - Cτ Q) / HJ. C(tp, ts) = tp w wᵀ + ts (the rest: s13, -2 mu66; shear
    # mu, (mu66)), w = (wx, [wx,] wz); C₁ = C(1 + τp(1+d2), 1 + τs(1+d2)), Cτ = C(τp, τs). X = ts₁ P - τs Q, X' = (1+d2) P - Q.
    function stress_adjoint!(b)
        P, e, Q, gr, Y = comps(Adj[b], sn), Gout[b], Qm[b], Grad[b], Yin[b]
        stress_adjoint_vti!(b, P, e, Q, gr, Y, true)
        if tilted_                     # P, Q, e in the frame of the axis: the VTI adjoint there, Y rotated back
            Pl, Ql, el = Loc[b]
            foreach_item!(backend, tti_adjoint_in!, (Pl, Ql, el, P, Q, e, angles(b), vpos), length(e[1]))
            stress_adjoint_vti!(b, Pl, el, Ql, gr, Y, false)
            B = blocks[b]
            foreach_item!(backend, tti_adjoint_out!, (Y, gr.tilt, gr.azim, Pl, Ql, el, P, Q, e, Mv[b], attenuation && !sh ? B.tp : nothing,
                                                     attenuation ? B.ts : nothing, d2, dtT, angles(b), vpos), length(e[1]))
        else
            stress_adjoint_vti!(b, P, e, Q, gr, Y, false)
        end
    end
    "the adjoint memory variables (`mem`: only those, from the adjoint stresses P) or the VTI adjoint of P, Q, e (`mem` false)"
    function stress_adjoint_vti!(b, P, e, Q, gr, Y, mem)
        wx, wz, s13, mu, m66 = Mv[b]
        pe, pP, pQ, Z = Wk[b]
        X, Xp = Xw[b], Xpw[b]
        B = blocks[b]
        τp, τs = attenuation ? (sh ? zero(FT) : B.tp, B.ts) : (zero(FT), zero(FT))
        if mem
            attenuation || return nothing
            for k in 1:nS
                fill!(Q[k], 0)
                for j in 1:NM
                    R = selectdim(Rb[b][k], 1, j)
                    @. R = c1[j] * R + dtT * d1[j] * P[k]
                    @. Q[k] += c2[j] * R
                end
            end
            return nothing
        end
        if sh                          # SH: σxy (C66) and σyz (C44), both shear (e: γ/2)
            for k in 1:2
                @. X[k] = (1 + τs * (1 + d2)) * P[k] - τs * Q[k]
                @. Xp[k] = (1 + d2) * P[k] - Q[k]
            end
            @. gr.m66 += dtT * X[1] * 2e[1]
            @. gr.mu += dtT * X[2] * 2e[2]
            attenuation && @. gr.ts += dtT * (m66 * Xp[1] * 2e[1] + mu * Xp[2] * 2e[2])
            @. Y[1] = m66 * X[1] / HJ[b]
            @. Y[2] = mu * X[2] / HJ[b]
            return nothing
        end
        w = N == 2 ? (wx, wz) : (wx, wx, wz)
        @. pe = w[1] * e[1]; @. pP = w[1] * P[1]; @. pQ = w[1] * Q[1]
        for k in 2:N
            @. pe += w[k] * e[k]; @. pP += w[k] * P[k]; @. pQ += w[k] * Q[k]
        end
        @. Z = (1 + τp * (1 + d2)) * pP - τp * pQ
        for k in 1:nS
            @. X[k] = (1 + τs * (1 + d2)) * P[k] - τs * Q[k]
            @. Xp[k] = (1 + d2) * P[k] - Q[k]
        end
        if N == 2
            @. gr.wx += dtT * (pe * ((1 + τp * (1 + d2)) * P[1] - τp * Q[1]) + e[1] * Z)
            @. gr.wz += dtT * (pe * ((1 + τp * (1 + d2)) * P[2] - τp * Q[2]) + e[2] * Z)
            @. gr.s13 += dtT * (X[1] * e[2] + X[2] * e[1])
            @. gr.mu += dtT * X[3] * 2e[3]
            attenuation && @. gr.ts += dtT * (s13 * (Xp[1] * e[2] + Xp[2] * e[1]) + mu * Xp[3] * 2e[3])
            @. Y[1] = (wx * Z + s13 * X[2]) / HJ[b]
            @. Y[2] = (wz * Z + s13 * X[1]) / HJ[b]
            @. Y[3] = mu * X[3] / HJ[b]
        else
            @. gr.wx += dtT * (pe * ((1 + τp * (1 + d2)) * (P[1] + P[2]) - τp * (Q[1] + Q[2])) + (e[1] + e[2]) * Z)
            @. gr.wz += dtT * (pe * ((1 + τp * (1 + d2)) * P[3] - τp * Q[3]) + e[3] * Z)
            @. gr.s13 += dtT * (X[1] * e[3] + X[3] * e[1] + X[2] * e[3] + X[3] * e[2])
            @. gr.m66 += dtT * (-2(X[1] * e[2] + X[2] * e[1]) + X[6] * 2e[6])
            @. gr.mu += dtT * (X[4] * 2e[4] + X[5] * 2e[5])
            attenuation && @. gr.ts += dtT * (s13 * (Xp[1] * e[3] + Xp[3] * e[1] + Xp[2] * e[3] + Xp[3] * e[2]) -
                                              2m66 * (Xp[1] * e[2] + Xp[2] * e[1]) + mu * (Xp[4] * 2e[4] + Xp[5] * 2e[5]) +
                                              m66 * Xp[6] * 2e[6])
            @. Y[1] = (wx * Z - 2m66 * X[2] + s13 * X[3]) / HJ[b]
            @. Y[2] = (wx * Z - 2m66 * X[1] + s13 * X[3]) / HJ[b]
            @. Y[3] = (wz * Z + s13 * (X[1] + X[2])) / HJ[b]
            @. Y[4] = mu * X[4] / HJ[b]; @. Y[5] = mu * X[5] / HJ[b]; @. Y[6] = m66 * X[6] / HJ[b]
        end
        attenuation && @. gr.tp += dtT * pe * ((1 + d2) * pP - pQ)
    end
    # 3. backward, segment by segment
    # forces at damped (boundary) nodes: the damping gradient needs v^{n+1/2} without the force (v new - its increment)
    damped(m) = (H = hostblocks[c.ftab.b[m]]; I = N == 2 ? CartesianIndex(c.ftab.i[m] + ox - c.ow, c.ftab.k[m]) :
                                                         CartesianIndex(c.ftab.i[m] + ox - c.ow, c.ftab.j[m] + oy - c.owy, c.ftab.k[m]);
                 any(k -> getfield(H, an[k])[I] != 0, 1:nV))
    forcedamp = any(damped, eachindex(c.ftab.b))
    Finc = forcedamp ? Tuple(ntuple(_ -> zero_(b), nV) for b in 1:nb) : nothing
    Fblk = forcedamp ? Tuple(merge(blocks[b], NamedTuple{vn}(Finc[b]), (; pml=nothing)) for b in 1:nb) : nothing
    store = [Tuple(ntuple(_ -> zero_(b), (3 + forcedamp)nV) for b in 1:nb) for _ in 1:K]   # per step: v old, v new, A σ, (v new without forces)
    pstore = pml ? [Tuple(map(similar, pml_arrays(blocks[b])) for b in 1:nb) for _ in 1:K] : nothing   # PML memory at each step
    for s0 in sort!(collect(keys(ckpt)); rev=true)
        s1 = min(s0 + K - 1, nt)
        for (b, B) in enumerate(blocks), (A, h) in zip(fields(B), ckpt[s0][b])
            copyto!(A, h)
        end
        for it in s0:s1
            q = store[it-s0+1]
            if pml
                for b in 1:nb
                    foreach(copyto!, pstore[it-s0+1][b], pml_arrays(blocks[b]))
                    foreach(copyto!, Tuple(getfield(scratch[b], k) for k in memnames(scratch[b])), pml_arrays(blocks[b]))
                end
            end
            Aop!(AblkF)
            for b in 1:nb, k in 1:nV
                copyto!(q[b][k], comps(blocks[b], vn)[k]); copyto!(q[b][2nV+k], Aout[b][k])
            end
            step!(it, false)
            for b in 1:nb, k in 1:nV
                copyto!(q[b][nV+k], comps(blocks[b], vn)[k])
            end
            if forcedamp
                foreach(F -> foreach(A -> fill!(A, 0), F), Finc)
                foreach_item!(backend, sh ? force_sh_cv! : N == 2 ? force_cv! : force3_cv!, (slim(Fblk, CV_FORCE), FS, window_step!(c.Wf, it), dtT), length(FS.seg) - 1)
                for b in 1:nb, k in 1:nV
                    q[b][3nV+k] .= q[b][nV+k] .- Finc[b][k]
                end
            end
        end
        for it in s1:-1:s0
            n = it - 1
            itw = n % record_every == 0 ? n ÷ record_every + 1 : 0
            q = store[it-s0+1]
            # moment tensors (stress glut, after the stress update)
            nM > 0 && run_once!(backend, moment_sens_cv!, (AM, Wmb.out[1], Adj, MSb, window_step!(Wmb, it), dtT, Val(sn)))
            c.sponge && foreach(b -> foreach(S -> S .*= blocks[b].gs, comps(Adj[b], sn)), 1:nb)      # (the sponge is diagonal)
            c.ntw > 0 && foreach_item!(backend, tf_grad_stress!, (Grad, Adj, slim(lite, (CV_METRIC..., :V1, :tilt, :azim)), Mv, c.TW, FT(c.tbeg + (n + 0.5) * c.dt), dtT, c.tfkind), c.ntw)
            # stress update: gradient (λ, μ, τ) and v̄ += dt Gᵀ (C₁ σ̄ - Cτ Q)
            for b in 1:nb, k in 1:nV
                copyto!(Win[b][k], q[b][nV+k])
                L === nothing || (L[b] .+= q[b][nV+k] .^ 2)
            end
            pml && for b in 1:nb
                foreach(copyto!, Tuple(getfield(scratch[b], k) for k in memnames(scratch[b])), pstore[it-s0+1][b])
            end
            Gop!(GblkP)
            foreach(stress_adjoint!, 1:nb)
            Aop!(Ablk)
            for b in 1:nb, k in 1:nV
                V = comps(Adj[b], vn)[k]
                @. V -= dtT * Hw[b] * Aout[b][k]
            end
            if pml                     # the stretched parts of G (stress update): Gᵀ of the slabs
                foreach((Y0, Y) -> foreach(copyto!, Y0, Y), Yin0, Yin)
                for Dp in PML
                    pml_adjoint!(Yin, Yin0, Dp.ms, Dp.pieces)
                    Aop!(Tuple(merge(Ablk[b], (; part=Dp.part)) for b in 1:nb))
                    for b in 1:nb, k in 1:nV
                        V = comps(Adj[b], vn)[k]
                        @. V -= dtT * Hw[b] * Aout[b][k]
                    end
                end
            end
            # recording at step n: v(tₙ) = (v^{n-1/2} + v^{n+1/2}) / 2
            itw > 0 && nst > 0 && run_once!(backend, residual!, (Adj, St, Rres, itw, FT(0.5)))
            # momentum v^{n+1/2} = P v^{n-1/2} + Q dt (1/ρJ) A σⁿ (+ forces): gradient (1/ρJ, damping) and σ̄ += Aᵀ(...)
            if nF > 0
                lf = window_step!(Wfb, it)
                run_once!(backend, force_grad!, (Grad, Adj, FSb, lf, dtT))
                run_once!(backend, force_sens_cv!, (AF, Wfb.out[1], Adj, slim(blocks, (:iρJ,)), FSb, lf, dtT, Val(vn)))
            end
            c.sponge && foreach(b -> foreach(V -> V .*= blocks[b].gs, comps(Adj[b], vn)), 1:nb)
            c.ntw > 0 && foreach_item!(backend, tf_grad_momentum!, (Grad, Adj, slim(lite, (CV_METRIC..., :V1)), c.TW, FT(c.tbeg + n * c.dt), dtT, c.tfkind), c.ntw)
            for b in 1:nb
                g, B = Grad[b], blocks[b]
                for (k, ab) in enumerate(comps(B, an))
                    V = comps(Adj[b], vn)[k]
                    @. g.iρJ += dtT * V * q[b][2nV+k] / (1 + ab * dtT / 2)
                    vn1 = q[b][(forcedamp ? 3nV : nV)+k]          # v^{n+1/2} of the momentum update: without forces,
                    if c.sponge                                # before the sponge factor
                        @. g.a[k] -= dtT / 2 * V * (q[b][k] + vn1 / B.gs) / (1 + ab * dtT / 2)
                    else
                        @. g.a[k] -= dtT / 2 * V * (q[b][k] + vn1) / (1 + ab * dtT / 2)
                    end
                    @. Win[b][k] = dtT * B.iρJ * V / ((1 + ab * dtT / 2) * Hw[b])
                end
            end
            Gop!()
            for b in 1:nb
                S = comps(Adj[b], sn)
                for k in 1:nS
                    @. S[k] -= HJ[b] * (1 + (k > nN)) * Gout[b][k]
                end
            end
            if pml                     # the stretched parts of A (momentum update): Aᵀ of the slabs
                foreach((W0, W) -> foreach(copyto!, W0, W), Win0, Win)
                for Dp in PML
                    pml_adjoint!(Win, Win0, Dp.mv, Dp.pieces)
                    Gop!(Tuple(merge(Gblk[b], (; part=Dp.part)) for b in 1:nb))
                    for b in 1:nb, k in 1:nS
                        S = comps(Adj[b], sn)[k]
                        @. S -= HJ[b] * (1 + (k > nN)) * Gout[b][k]
                    end
                end
            end
            for b in 1:nb
                for (k, ab) in enumerate(comps(blocks[b], an))
                    V = comps(Adj[b], vn)[k]
                    @. V *= (1 - ab * dtT / 2) / (1 + ab * dtT / 2)
                end
            end
            itw > 0 && nst > 0 && run_once!(backend, residual!, (Adj, St, Rres, itw, FT(0.5)))
        end
    end
    KernelAbstractions.synchronize(backend)
    # 4. to ρ, vp, vs [per kg/m³, km/s], (ε, δ, γ,) Qp, Qs on the nodes of the model (owned nodes; summed over the processes)
    aniso = has_anisotropy(model)
    gρ, gvp, gvs, gε, gδ, gγ, gqp, gqs, gL, gθ, gφ = (zeros(nx, ny, nzt) for _ in 1:11)
    f = model.fields
    for (b, H) in enumerate(hostblocks)
        g = map(x -> x isa Tuple ? map(A -> Float64.(Array(A)), x) : x === nothing ? nothing : Float64.(Array(x)), Grad[b])     # (loop-local)
        hf = (; rho=H.ρ, vp=H.vp, vs=H.vs, ε=H.ε, δ=H.δ, γ=H.γ, taup=H.taup, taus=H.taus)
        own = N == 2 ? CartesianIndices((hx+1:hx+c.no, 1:H.nz)) : CartesianIndices((hx+1:hx+c.nox, hx+1:hx+c.noy, 1:H.nz))
        off = N == 2 ? CartesianIndex(ox, 0) : CartesianIndex(ox, oy, 0)                     # local -> global
        offw = N == 2 ? CartesianIndex(ox - c.ow, 0) : CartesianIndex(ox - c.ow, oy - c.owy, 0)   # local -> window
        Lh = L === nothing ? nothing : Float64.(Array(L[b]))
        r0 = H.kmap[CartesianIndex(ntuple(_ -> 1, N - 1)..., 2)] - 2           # the mesh row of the block's node k: r0 + k
        for Il in own
            Ig, I = Il + off, Il + offw
            node = N == 2 ? CartesianIndex(Ig[1], 1, H.kmap[I]) : CartesianIndex(Ig[1], Ig[2], H.kmap[I])
            gm = (g.wx[Il], g.wz[Il], g.s13[Il], g.mu[Il], g.m66[Il])
            # directions (ρ, vp, vs, ε, δ, γ, τp, τs) through the moduli; the damping (vp, vs, ε, γ); 1/(ρJ)
            dir(j) = (t = vti_tangent(hf, I, ntuple(q -> Float64(q == j), 8), ts, fq_ref); sum(gm[q] * t[q] for q in 1:5))
            da(q) = sum(k -> g.a[k][Il] * H.da[k][q][I], 1:nV)
            gρ[node] += dir(1) - g.iρJ[Il] / (H.ρ[I]^2 * H.J[I])
            gvp[node] += dir(2) + da(1)
            gvs[node] += dir(3) + da(2)
            if aniso
                gε[node] += dir(4) + da(3); gδ[node] += dir(5); gγ[node] += dir(6) + da(4)
            end
            if tilted_                                 # [per degree]
                gθ[node] += deg2rad(g.tilt[Il])
                tilt3 && (gφ[node] += deg2rad(g.azim[Il]))
            end
            if attenuation
                dtp, dts = g.tp[Il] + dir(7), g.ts[Il] + dir(8)
                qp, qs = haskey(f, :Qp) ? f.Qp[node] : Inf, haskey(f, :Qs) ? f.Qs[node] : Inf
                gqp[node] += isfinite(qp) ? -dtp * H.taup[I] / qp : 0.0       # tau = NM zeta / Q
                gqs[node] += isfinite(qs) ? -dts * H.taus[I] / qs : 0.0
            end
            if Lh !== nothing && (b == 1 || I[N] > 1)      # (on the mesh nodes; interface rows from the block below, as snapshots)
                gL[CartesianIndex(Base.front(Tuple(node))..., r0 + I[N])] += Lh[Il]
            end
        end
    end
    par === nothing || foreach(A -> MPI.Allreduce!(A, +, par.comm), (gρ, gvp, gvs, gε, gδ, gγ, gqp, gqs, gL, gθ, gφ))
    rho_unit = density_unit === :g_cm3 ? 1e3 : 1.0
    ill = L === nothing ? (;) : (; illumination=gL)
    vti = merge(aniso ? (; epsilon=gε, delta=gδ, gamma=gγ) : (;), tilted_ ? (tilt3 ? (; tilt=gθ, azimuth=gφ) : (; tilt=gθ)) : (;))
    grad = CartData(model.x.val, model.y.val, model.z.val, merge((; rho=gρ .* rho_unit, vp=gvp .* 1e3, vs=gvs .* 1e3), vti, (; Qp=gqp, Qs=gqs), ill))
    flush!(Wfb); flush!(Wmb)
    return J, grad, curvilinear_source_gradients(c, sh ? 1 : N, AF, gmF, AM, gmM)
end

"""
The source gradients of [`misfit_gradient`](@ref) from the sensitivities per table entry (`AF`, `AM`) and per step (`gmF`,
`gmM`: ∂J/∂ moment rate). Location: from the derivatives of the interpolation weights (central differences of the tables,
shifted by 1 mm).
"""
function curvilinear_source_gradients(c, N, AF, gmF, AM, gmM)
    sh = N == 1                                         # (SH: N = 1; the dimension is then 2)
    N = max(N, 2)
    fcs = sh ? (2,) : N == 2 ? (1, 3) : (1, 2, 3)       # the force and moment tensor components that are injected
    mcs = sh ? (6, 4) : N == 2 ? (1, 3, 5) : 1:6
    (; par, dt, nt, tbeg, stf, forces, moments, srcs, ftab, mtab) = c
    sum_(A) = (B = Float64.(Array(A)); par === nothing || MPI.Allreduce!(B, +, par.comm); B)
    AF, AM = Float64.(Array(AF)), Float64.(Array(AM))          # (per local entry)
    gmF, gmM = sum_(gmF), sum_(gmM)
    key(t, m) = N == 2 ? (t.b[m], t.i[m], t.k[m], t.n[m]) : (t.b[m], t.i[m], t.j[m], t.k[m], t.n[m])
    # ∂J/∂(weight of entry m) per source; the weights' derivatives with respect to x, (y,) z
    wf = [sum(forces[ftab.n[m]].F[fc] * AF[m, q] for (q, fc) in enumerate(fcs)) for m in eachindex(ftab.n)]
    wm = [-sum(moments[mtab.n[m]].M0 * moments[mtab.n[m]].M[mc] * AM[m, q] for (q, mc) in enumerate(mcs))
          for m in eachindex(mtab.n)]
    h = 1e-6
    dloc = map(1:N) do a
        δ(s) = ntuple(b -> b == a ? s * h : 0.0, N)
        (fp, mp), (fm, mm) = c.tables(δ(1)), c.tables(δ(-1))
        dw(t, tp, tm, w, ns) = (G = zeros(ns);
            cp = Dict(key(tp, m) => tp.c[m] for m in eachindex(tp.c)); cm = Dict(key(tm, m) => tm.c[m] for m in eachindex(tm.c));
            for m in eachindex(t.c)
                G[t.n[m]] += w[m] * (get(cp, key(t, m), 0.0) - get(cm, key(t, m), 0.0)) / 2h
            end; G)      # shortcut: nodes entering the support at the shifted position are left out (weights ~0 there)
        (sum_(dw(ftab, fp, fm, wf, length(forces))), sum_(dw(mtab, mp, mm, wm, length(moments))))
    end
    xyz(l, q) = N == 2 ? (dloc[1][q][l], 0.0, dloc[2][q][l]) : (dloc[1][q][l], dloc[2][q][l], dloc[3][q][l])
    # ∂J/∂t0, ∂J/∂trise from the gradient with respect to the moment rate (central differences of the source time function)
    function timing(gm, times, src)
        source_stf(src, stf) === :delta && return NaN, NaN        # (an impulse: no derivative with respect to t0, trise)
        mr(t, t0, tr) = moment_rate(source_stf(src, stf), t, t0, tr)
        e = 1e-6 * src.trise
        dt0 = sum(gm[n] * (mr(times[n], src.t0 + e, src.trise) - mr(times[n], src.t0 - e, src.trise)) / 2e for n in 1:nt)
        dtr = sum(gm[n] * (mr(times[n], src.t0, src.trise + e) - mr(times[n], src.t0, src.trise - e)) / 2e for n in 1:nt)
        return dt0, dtr
    end
    tf = [tbeg + (n - 1) * dt for n in 1:nt]                     # times of the force injections
    tm = [tbeg + (n - 0.5) * dt for n in 1:nt]                   # and of the moment tensors
    entries(t, l) = findall(==(l), t.n)
    fs = map(enumerate(forces)) do (l, src)
        e = entries(ftab, l)
        dF = sum_([sum(ftab.c[m] * AF[m, q] for m in e; init=0.0) for q in eachindex(fcs)])
        x, y, z = xyz(l, 1)
        t0, trise = timing(gmF[l, :], tf, src)
        (; F=ntuple(j -> (q = findfirst(==(j), fcs); q === nothing ? 0.0 : dF[q]), 3), x, y, z, t0, trise, stf=gmF[l, :], t=tf)
    end
    mts = map(enumerate(moments)) do (l, src)
        e = entries(mtab, l)
        dM = sum_([-sum(mtab.c[m] * AM[m, q] for m in e; init=0.0) for q in eachindex(mcs)])
        dP = ntuple(j -> (q = findfirst(==(j), mcs); q === nothing ? 0.0 : dM[q]), 6)      # ∂J/∂(M0 M)
        x, y, z = xyz(l, 2)
        t0, trise = timing(gmM[l, :], tm, src)
        (; M=dP, M0=sum(src.M[j] * dP[j] for j in 1:6), x, y, z, t0, trise, stf=gmM[l, :], t=tm)
    end
    return map(srcs) do src
        src isa Union{MomentTensorSource,ForceSource} || return nothing        # (a teleseismic wave)
        i = src isa MomentTensorSource ? findfirst(x -> x === src, moments) : findfirst(x -> x === src, forces)
        src isa MomentTensorSource ? mts[i] : fs[i]
    end
end

# ------------------------------------------------------------------ Born modelling (2D and 3D)

"""
Born modelling on curvilinear grids: the simulation runs with Born numbers (`T = Tan{…}`); the medium arrays of the blocks get
the tangents of the perturbation δm (the forward chain rule of the gradient's, through `kmap`: the model row of each node).
"""
function born_curvilinear(d::BornDriver, c)
    (; blocks, hostblocks, model, density_unit, attenuation, ts, fq_ref, loc, nt, backend) = c
    F0 = typeof(value(zero(c.FT)))
    f, δf = model.fields, d.δm.fields
    rho_unit = density_unit === :g_cm3 ? 1e3 : 1.0
    for (B, H) in zip(blocks, hostblocks)
        node(I) = ndims(H.kmap) == 2 ? CartesianIndex(I[1] + c.ow, 1, H.kmap[I]) : CartesianIndex(I[1] + c.ow, I[2] + c.owy, H.kmap[I])
        at(k, s, z) = haskey(δf, k) ? [s * Float64(δf[k][node(I)]) for I in CartesianIndices(H.kmap)] : fill(z, size(H.kmap))
        δρ, δvp, δvs = at(:rho, rho_unit, 0.0), at(:vp, 1e3, 0.0), at(:vs, 1e3, 0.0)
        δε, δδ, δγ = at(:epsilon, 1.0, 0.0), at(:delta, 1.0, 0.0), at(:gamma, 1.0, 0.0)
        δqp, δqs = at(:Qp, 1.0, 0.0), at(:Qs, 1.0, 0.0)
        qp = haskey(f, :Qp) ? [Float64(f.Qp[node(I)]) for I in CartesianIndices(H.kmap)] : fill(Inf, size(H.kmap))
        qs = haskey(f, :Qs) ? [Float64(f.Qs[node(I)]) for I in CartesianIndices(H.kmap)] : fill(Inf, size(H.kmap))
        ρ, vp, vs = H.ρ, H.vp, H.vs
        δtp = @. ifelse(isfinite(qp), -H.taup / qp * δqp, 0.0)        # tau = NM zeta / Q
        δts = @. ifelse(isfinite(qs), -H.taus / qs * δqs, 0.0)
        δ = Dict{Symbol,Array{Float64}}(:iρJ => -δρ ./ (ρ .^ 2 .* H.J))
        if haskey(B, :wx) || haskey(B, :S1)        # VTI (and SH: mu, mu66): the tangents of the stored moduli
            hf = (; rho=H.ρ, vp=H.vp, vs=H.vs, ε=H.ε, δ=H.δ, γ=H.γ, taup=H.taup, taus=H.taus)
            dx = (δρ, δvp, δvs, δε, δδ, δγ, δtp, δts)
            T = map(I -> vti_tangent(hf, I, map(A -> A[I], dx), ts, fq_ref), CartesianIndices(H.kmap))
            for (q, k) in enumerate((:wx, :wz, :s13, :mu, :mu66))
                haskey(B, k) && (δ[k] = getindex.(T, q))
            end
        else
            χp, χs = visco_chi.((ts,), H.taup, fq_ref), visco_chi.((ts,), H.taus, fq_ref)
            dχp, dχs = visco_dchi.((ts,), H.taup, fq_ref), visco_dchi.((ts,), H.taus, fq_ref)
            δ[:mu] = @. δρ * vs^2 / χs^2 + 2ρ * vs * δvs / χs^2 - 2ρ * vs^2 / χs^3 * dχs * δts
            δ[:lam] = @. δρ * vp^2 / χp^2 + 2ρ * vp * δvp / χp^2 - 2ρ * vp^2 / χp^3 * dχp * δtp - 2δ[:mu]
        end
        haskey(B, :tilt) && (δ[:tilt] = deg2rad.(at(:tilt, 1.0, 0.0)))     # TTI: the angles of the axis [rad]
        haskey(B, :azim) && (δ[:azim] = deg2rad.(at(:azimuth, 1.0, 0.0)))
        if attenuation
            haskey(B, :tp) && (δ[:tp] = δtp)            # (SH: no P waves)
            δ[:ts] = δts
        end
        # impedance damping: da[c] = (∂vp, ∂vs, ∂ε, ∂γ) of a_c
        for (m, d) in enumerate(H.da)
            δ[Symbol(:a, m)] = @. d[1] * δvp + d[2] * δvs + d[3] * δε + d[4] * δγ
        end
        for (k, δk) in δ
            A = getfield(B, k)
            copyto!(A, Tan.(value.(Array(A)), F0.(loc(δk))))
        end
    end
    for it in 1:nt
        c.step!(it, true)
    end
    KernelAbstractions.synchronize(backend)
    return c.seismograms(c.all_traces(value)), c.seismograms(c.all_traces(tangent))
end
drive_curvilinear(d::BornDriver, c) = born_curvilinear(d, c)
drive_curvilinear3(d::BornDriver, c) = born_curvilinear(d, c)
