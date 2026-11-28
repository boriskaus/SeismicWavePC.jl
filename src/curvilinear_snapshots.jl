# Snapshots (netCDF planes, OpenSWPC format: `Snapshots`) on curvilinear grids. The points are on the node columns of the
# mesh (decimated): the free surface `fs` is the top node of every column (the true topography); horizontal planes `xy` and
# the sections `xz`, `yz` (2D: `xz`, the model) are interpolated along the columns (Lagrange in η, 6 nodes) at regular
# heights, and are zero above the topography (medium: NaN). Every step a kernel accumulates the displacement and the peak
# values; the files are written as on the staggered grid (by rank 0, or in pieces by every process: `write_planes`).

"""
Velocity (Lagrange in η), displacement and peaks at plane point `n`; the row of `B` at snapshot steps, with the divergence and
rotation of the node fields `D` (per block: div, rot (2D) or div, rot_x, rot_y, rot_z; `nothing` without `:ps`). (They are
stored × 1e3: the writer of the staggered grid converts its 1/km to 1/s.)
"""
Base.@propagate_inbounds function plane_cv!(B, U, PK, blocks, D, P, dt, store, ::Val{N}, n) where {N}
    b = P.b[n]
    T = eltype(U)
    v = ntuple(_ -> zero(T), Val(N))
    if b > 0
        Bk = blocks[b]
        Vs = N == 1 ? (Bk.V1,) : N == 2 ? (Bk.V1, Bk.V2) : (Bk.V1, Bk.V2, Bk.V3)
        for q in 1:6
            w = P.w[q, n]
            I = N <= 2 ? CartesianIndex(P.i[n], P.k0[n] + q - 1) : CartesianIndex(P.i[n], P.j[n], P.k0[n] + q - 1)
            v = v .+ map(A -> w * A[I], Vs)
        end
    end
    u = ntuple(c -> U[n, c] + v[c] * dt, Val(N))
    for c in 1:N
        U[n, c] = u[c]
    end
    vh = N == 3 ? sqrt(v[1]^2 + v[2]^2) : abs(v[1])
    peaks!(PK, n, v[N], vh, sqrt(sum(abs2, v)))
    if store
        if D === nothing
            store_row!(B, n, (v..., u...))
        else
            T = eltype(U)
            q = ntuple(_ -> zero(T), Val(2N - 2))
            if b > 0
                for c in 1:6
                    w = P.w[c, n]
                    I = N == 2 ? CartesianIndex(P.i[n], P.k0[n] + c - 1) : CartesianIndex(P.i[n], P.j[n], P.k0[n] + c - 1)
                    q = q .+ map(A -> w * A[I], D[b])
                end
            end
            store_row!(B, n, (v..., u..., (q .* T(1e3))...))
        end
    end
end

"""
The divergence and rotation of the velocities at the nodes, for `:ps` snapshots: the strain operator of the adjoint (the
stress kernel with λ = 0, μ = ½) on the velocities with one component negated turns a shear strain into a rotation:
2D (vx, -vz): div = e₁₁ - e₂₂, rot = ∂z vx - ∂x vz = 2e₁₂; 3D (vx, vy, -vz): div = e₁₁ + e₂₂ - e₃₃, rot_x = -2e₂₃,
rot_y = 2e₁₃, and (vx, -vy, vz): rot_z = -2e₁₂. Returns (the fields per block, a function that computes them).
"""
function divrot_cv(blocks, stress!, N, backend, FT)
    zero_(B) = KernelAbstractions.zeros(backend, FT, size(B.J))
    half(B) = (A = zero_(B); fill!(A, FT(0.5)); A)
    vn, sn = N == 2 ? ((:V1, :V2), (:S11, :S22, :S12)) : ((:V1, :V2, :V3), (:S11, :S22, :S33, :S23, :S13, :S12))
    rn = N == 2 ? (:R11, :R22, :R12) : (:R11, :R22, :R33, :R23, :R13, :R12)
    plain(B, V, S) = merge(Base.structdiff(B, NamedTuple{(:wx, :wz, :s13, :mu66, :tilt, :azim)}), NamedTuple{vn}(V), NamedTuple{sn}(S),
                           NamedTuple{rn}(ntuple(_ -> nothing, length(rn))), (; lam=zero_(B), mu=half(B), tp=nothing, ts=nothing, pml=nothing))
    neg = Tuple(zero_(B) for B in blocks)
    SA = Tuple(ntuple(_ -> zero_(B), length(sn)) for B in blocks)
    SB = N == 3 ? Tuple(ntuple(_ -> zero_(B), 6) for B in blocks) : nothing
    opA = Tuple(plain(B, N == 2 ? (B.V1, neg[b]) : (B.V1, B.V2, neg[b]), SA[b]) for (b, B) in enumerate(blocks))
    opB = N == 3 ? Tuple(plain(B, (B.V1, neg[b], B.V3), SB[b]) for (b, B) in enumerate(blocks)) : nothing
    D = Tuple(ntuple(_ -> zero_(B), 2N - 2) for B in blocks)
    function compute!()
        for (b, B) in enumerate(blocks)
            neg[b] .= .-(N == 2 ? B.V2 : B.V3)
            foreach(A -> fill!(A, 0), SA[b])
        end
        stress!(nothing, one(FT), opA)
        for b in eachindex(blocks)
            S = SA[b]
            if N == 2
                @. D[b][1] = S[1] - S[2]; @. D[b][2] = 2S[3]
            else
                @. D[b][1] = S[1] + S[2] - S[3]; @. D[b][2] = -2S[4]; @. D[b][3] = 2S[5]
            end
        end
        if N == 3
            for (b, B) in enumerate(blocks)
                neg[b] .= .-B.V2
                foreach(A -> fill!(A, 0), SB[b])
            end
            stress!(nothing, one(FT), opB)
            foreach(b -> (@. D[b][4] = -2SB[b][6]), eachindex(blocks))
        end
        KernelAbstractions.synchronize(backend)
    end
    return D, compute!
end

"""
The planes of `s` on a curvilinear grid (`N` = 2, 3): points, device buffers and files, as `setup_planes` of the staggered
grid. `c`: the setup of the simulation (hostblocks, the node coordinates `gx`, `gy` [m], the nodes `ia:ib`, `ja:jb` of this
process and their offsets).
"""
function setup_planes_cv(s::Snapshots, Nc, c)
    # (Nc: the velocity components, 1 SH, 2 P-SV, 3 3D; N: the dimension)
    N = max(Nc, 2)
    (; hostblocks, nx, ny, gx, gy, Zm, ia, ib, ja, jb, ox, oy, dev, backend, FT, dir, me, parallel, par, ow, owy) = c
    # (host arrays: the columns of the window of this process, index - (ow, owy); the points of other processes: theirs)
    N == 2 && (s.xy !== nothing || s.yz !== nothing || s.fs || s.ob) && error("2D snapshots: only xz")
    s.ob && error("curvilinear grids: no ocean bottom (solids only)")
    solver = Nc == 3 ? "3d" : Nc == 1 ? "sh" : "psv"
    nc = Nc
    # decimated nodes per axis, regular heights (from the top of the mesh down, with the mean row spacing)
    dec = s.decimate
    xi, yi = collect(1:dec[1]:nx), collect(1:dec[2]:ny)
    ztop, zbot = cv_max(par, maximum(Zm)), -cv_max(par, -minimum(Zm))
    dz = (ztop - zbot) / (size(Zm, ndims(Zm)) - 1) * dec[3]
    zl = collect(range(ztop, zbot; step=-dz))                                 # [m], z up
    "(block, k0, weights in η) of height z [m] in column (i, j) of the mesh, or nothing above / below it"
    function column(i, j, z)
        for (b, H) in enumerate(hostblocks)
            col = N == 2 ? H.Z[i-ow, :] : H.Z[i-ow, j-owy, :]
            (col[1] - 1e-6 <= z <= col[end] + 1e-6) || continue
            gη, _ = sbp_grid(0.0, 1.0, H.nz)
            zof(η) = let (q0, wq) = lagrange_weights(gη, η)
                sum(wq[c] * col[q0+c-1] for c in 1:6)
            end
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                zof(mid) < z ? (lo = mid) : (hi = mid)
            end
            k0, w = lagrange_weights(gη, (lo + hi) / 2)
            return (b, k0, w)
        end
        return nothing
    end
    top(i, j) = (nb = length(hostblocks); nz = hostblocks[nb].nz; (nb, nz - 5, [0.0, 0, 0, 0, 0, 1]))
    nearest(g, v) = argmin(abs.(g .- 1e3v))
    inx(i) = ia <= i <= ib
    iny(j) = N == 2 || ja <= j <= jb
    # planes: name, the point of every (axis 1, axis 2) index pair, the global indices along the axes and their coordinates
    planes = []
    km(g, ii) = g[ii] ./ 1e3
    if N == 3
        s.fs && push!(planes, (:fs, (i, j) -> top(i, j), (xi, yi), ("x" => km(gx, xi), "y" => km(gy, yi)), (i, j) -> (i, j)))
        if s.xy !== nothing
            z = 1e3s.xy
            push!(planes, (:xy, (i, j) -> column(i, j, z), (xi, yi), ("x" => km(gx, xi), "y" => km(gy, yi)), (i, j) -> (i, j)))
        end
        if s.xz !== nothing
            j0 = nearest(gy, s.xz)
            push!(planes, (:xz, (i, l) -> column(i, j0, zl[l]), (xi, eachindex(zl)), ("x" => km(gx, xi), "z" => -zl ./ 1e3), (i, l) -> (i, j0)))
        end
        if s.yz !== nothing
            i0 = nearest(gx, s.yz)
            push!(planes, (:yz, (j, l) -> column(i0, j, zl[l]), (yi, eachindex(zl)), ("y" => km(gy, yi), "z" => -zl ./ 1e3), (j, l) -> (i0, j)))
        end
    else
        s.xz === nothing || push!(planes, (:xz, (i, l) -> column(i, 1, zl[l]), (xi, eachindex(zl)), ("x" => km(gx, xi), "z" => -zl ./ 1e3), (i, l) -> (i, 1)))
    end
    isempty(planes) && error("Snapshots without planes")
    fields = snapshot_fields(Nc == 3 ? :p3d : Nc == 1 ? :sh : :psv)
    map(planes) do (name, point, (A1, A2), axes, col)
        # the points (in pieces: those of the columns of this process; a piece is a block of the plane)
        mine(a1, a2) = (cc = col(a1, a2); inx(cc[1]) && iny(cc[2]))
        if s.pieces
            I1 = [a for a in A1 if any(a2 -> mine(a, a2), A2)]
            I2 = [a for a in A2 if any(a1 -> mine(a1, a), A1)]
            whole = (length(A1), length(A2))
            offset = (count(<(minimum(I1; init=typemax(Int))), A1), count(<(minimum(I2; init=typemax(Int))), A2))
            isempty(I1) || isempty(I2) ? (I1 = I2 = Int[]) : nothing
        else
            I1, I2 = A1, A2
        end
        pairs = vec([(a1, a2) for a1 in I1, a2 in I2])
        own = [n for (n, (a1, a2)) in enumerate(pairs) if mine(a1, a2)]
        isown = falses(length(pairs)); isown[own] .= true
        pts = [isown[n] ? point(a1, a2) : nothing for (n, (a1, a2)) in enumerate(pairs)]
        cols = [col(a1, a2) for (a1, a2) in pairs]
        m = max(length(own), 1)
        P = (; b=dev(Int32[pts[n] === nothing ? 0 : pts[n][1] for n in own]), i=dev(Int32[cols[n][1] - ox for n in own]),
             j=dev(Int32[cols[n][2] - oy for n in own]), k0=dev(Int32[pts[n] === nothing ? 1 : pts[n][2] for n in own]),
             w=dev(FT[pts[n] === nothing ? 0.0 : pts[n][3][q] for q in 1:6, n in own]))
        shape = (length(I1), length(I2))
        # the medium at the points (NaN above the topography)
        function med(f)                  # (of the points of this process; collected by rank 0 unless in pieces)
            v = [!isown[n] ? 0.0 : p === nothing ? NaN : (H = hostblocks[p[1]]; A = getfield(H, f);
                     sum(p[3][q] * (N == 2 ? A[cc[1]-ow, p[2]+q-1] : A[cc[1]-ow, cc[2]-owy, p[2]+q-1]) for q in 1:6))
                 for (n, (p, cc)) in enumerate(zip(pts, cols))]
            s.pieces || par === nothing || MPI.Allreduce!(v, +, par.comm)
            return v
        end
        medium = (rho=med(:ρ), lambda=med(:lam), mu=med(:mu))
        files = Dict{Symbol,String}()
        if s.pieces ? !isempty(pairs) : me == 0
            ax = s.pieces ? (axes[1][1] => axes[1][2][indexin(I1, A1)], axes[2][1] => axes[2][2][indexin(I2, A2)]) : axes
            for q in s.quantities
                vars, unit, _ = fields[q]
                file = joinpath(mkpath(dir), "$(s.title).$solver.$name.$q" * (s.pieces ? "." * lpad(me, 4, '0') : "") * ".nc")
                snapshot_create(file, ax, vars, unit, map(A -> reshape(A, shape), medium); piece=s.pieces ? (whole, offset) : nothing)
                files[q] = file
            end
        end
        (; name, own, npoints=length(pairs), shape, pts=P, files, pieces=s.pieces,
           B=KernelAbstractions.zeros(backend, FT, m, 2nc + (:ps in s.quantities) * (2nc - 2)), U=KernelAbstractions.zeros(backend, FT, m, nc),
           PK=KernelAbstractions.zeros(backend, FT, m, 3))
    end
end
