# The Matterhorn model of Limberger et al. (2025, Sci. Rep. 15, 23914; doi:10.1038/s41598-025-08218-5), shared by the
# 3D and 2D scripts. Lengths in km (x east, y north, z up, origin at the summit), velocities in km/s.
#
# - Topography: swissALTIRegio (10 m), Gaussian-smoothed (σ = 40 m), real within 5 km of the summit (the 10 km × 10 km of
#   the paper) and blended (cosine) to a flat surface at the mean height of 2.95 km between 5 and 6.5 km: the plane wave
#   is injected along a box (total-field / scattered-field, `TeleseismicWave`), whose sides must lie in flat, laterally
#   homogeneous ground.
# - Material: vp from 3 km/s at the summit (4.478 km) linearly to 6 km/s at -2 km (constant below), vs = vp/1.7, density
#   2600-3000 kg/m³ in the same way (the colour scale of Fig. 1 of the paper), Qp = 200, Qs = 100. Q fades out towards the
#   box boundary (5.5-6.5 km, -1 to -1.6 km), which must be elastic.
include("dem.jl")

const Z_TOP, Z_BOT = 4.478, -2.0                                  # the gradient: summit and bottom of the paper's model
const Z_FLAT = 2.95                                               # the flat surface around the mountain
const R_DEM, R_FLAT = 5.0, 6.5                                    # real topography inside R_DEM, flat outside R_FLAT
const BOX = 7.0                                                   # injection box: |x|, |y| <= BOX, z >= Z_BOT

vp_of(z) = 3.0 + 3.0 * (Z_TOP - clamp(z, Z_BOT, Z_TOP)) / (Z_TOP - Z_BOT)
rho_of(z) = 2600.0 + 400.0 * (Z_TOP - clamp(z, Z_BOT, Z_TOP)) / (Z_TOP - Z_BOT)
cosramp(r, a, b) = r <= a ? 1.0 : r >= b ? 0.0 : (1 + cos(π * (r - a) / (b - a))) / 2
"1/Q weight: 1 in the mountain, 0 (elastic) within 6 nodes (spacing dx) of the box boundary."
qweight(x, y, z, dx) = cosramp(max(abs(x), abs(y)), BOX - 6dx - 1.0, BOX - 6dx) * cosramp(-z, -Z_BOT - 6dx - 0.6, -Z_BOT - 6dx)

"Separable Gaussian smoothing (σ in grid points) of a matrix."
function gauss_smooth(Z, σ)
    r = ceil(Int, 3σ)
    w = exp.(-((-r:r) ./ σ) .^ 2 ./ 2)
    w ./= sum(w)
    conv1(A, dim) = mapslices(v -> [sum(w[k+r+1] * v[clamp(i + k, 1, length(v))] for k in -r:r) for i in eachindex(v)], A; dims=dim)
    return conv1(conv1(Z, 1), 2)
end

"The surface z(x, y) [km]: the smoothed DEM, blended to Z_FLAT outside R_DEM. `flat = true`: Z_FLAT everywhere (tests)."
function matterhorn_surface(; flat=false, σ=0.04)
    flat && return (x, y) -> Z_FLAT
    d = matterhorn_dem()
    Zs = gauss_smooth(d.z, σ / 0.01)
    x0, y0, h = d.x[1], d.y[1], 0.01
    function topo(x, y)
        w = cosramp(max(abs(x), abs(y)), R_DEM, R_FLAT)
        w == 0 && return Z_FLAT
        fi, fj = (x - x0) / h + 1, (y - y0) / h + 1
        i, j = clamp(floor(Int, fi), 1, size(Zs, 1) - 1), clamp(floor(Int, fj), 1, size(Zs, 2) - 1)
        s, t = fi - i, fj - j
        z = (1 - s) * (1 - t) * Zs[i, j] + s * (1 - t) * Zs[i+1, j] + (1 - s) * t * Zs[i, j+1] + s * t * Zs[i+1, j+1]
        return w * z + (1 - w) * Z_FLAT
    end
    return topo
end

"""
Terrain-following mesh with flat rows below `zb` [km] (the injection box needs horizontally layered material along its
bottom), bending smoothly (C2 smoothstep) up to the surface `topo` above: the rows of a flat mesh (surface Z_FLAT, row
spacing `dz`) are lifted by (topo - Z_FLAT) · s(u), u = (z - zb)/(Z_FLAT - zb). 3D: `y` a range; 2D: `y = nothing`.
"""
function matterhorn_mesh(x, y, topo; dz, bottom, zb=-1.5)
    flat = y === nothing ? curvilinear_mesh(x, [_ -> Z_FLAT]; dz, bottom) : curvilinear_mesh(x, y, [(_, _) -> Z_FLAT]; dz, bottom)
    X, Y, Zf = flat.x.val, flat.y.val, flat.z.val
    s(u) = u <= 0 ? 0.0 : u >= 1 ? 1.0 : u^3 * (10 - 15u + 6u^2)
    T = y === nothing ? topo.(X[:, :, end], 0.0) : topo.(X[:, :, end], Y[:, :, end])
    Z = [Zf[i, j, k] + (T[i, j] - Z_FLAT) * s((Zf[i, j, k] - zb) / (Z_FLAT - zb)) for i in axes(Zf, 1), j in axes(Zf, 2), k in axes(Zf, 3)]
    all(diff(Z; dims=3) .> 0) || error("matterhorn_mesh: rows cross (valleys too deep for zb)")
    return CartData(X, Y, Z, (; z=copy(Z)))
end

"""
The 2D north-south section through the summit (x pointing north): mesh, model and the surface height `sec(x)` [km], for
the node spacing `dx` and `na` PML nodes.
"""
function section_model(dx, na)
    topo3 = matterhorn_surface()
    sec(x) = topo3(0.0, x)
    L = BOX + (na + SeismicWavePC.SBP_NB + 10) * dx
    bottom = Z_BOT - (na + SeismicWavePC.SBP_NB + 12) * 0.75dx
    mesh = matterhorn_mesh(range(-L, L, length=round(Int, 2L / dx) + 1), nothing, (x, _) -> sec(x); dz=0.75dx, bottom)
    return mesh, matterhorn_model(mesh, dx; section=true), sec
end

"The Ormsby wavelet of the paper as the particle velocity of a plane wave: centred at 5 s, tapered to 0 at 0 and 10 s."
function ormsby_stf()
    t = collect(0:0.002:10)
    return SampledSTF(t, ormsby(t .- 5) .* cosramp.(abs.(t .- 5), 4.0, 5.0))
end

"Fill the material of a curvilinear mesh (2D: y = 0 is the N-S section, its x axis pointing north)."
function matterhorn_model(mesh, dx; section=false)
    X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
    vp = vp_of.(Z)
    w = section ? qweight.(0.0, X, Z, dx) : qweight.(X, Y, Z, dx)
    Qp, Qs = 200.0 ./ w, 100.0 ./ w                               # Inf where w = 0: elastic
    return addfield(mesh, (; rho=rho_of.(Z), vp, vs=vp ./ 1.7, Qp, Qs))
end

"""
Zero-phase Ormsby wavelet with the corner frequencies f = (f1, f2, f3, f4) [Hz] (flat between f2 and f3), at the times t:
the source of the paper (flat between 0.3 and 3.5 Hz, Fig. S1).
"""
function ormsby(t, f=(0.2, 0.3, 3.5, 3.65))
    f1, f2, f3, f4 = f
    s(a, τ) = π * a^2 * (τ == 0 ? 1.0 : (sin(π * a * τ) / (π * a * τ))^2)
    return [(s(f4, τ) - s(f3, τ)) / (f4 - f3) - (s(f2, τ) - s(f1, τ)) / (f2 - f1) for τ in t]
end
ricker(t, f0) = [(1 - 2(π * f0 * τ)^2) * exp(-(π * f0 * τ)^2) for τ in t]

"""
The seismograms `v[it, …]` (sample interval dt) of a source with the moment rate `:gaussian` (rise time tr, origin 0)
converted to those of the wavelet `w(τ)` (a function of the times τ, centred at τ = 0) delayed by `delay`: the ratio of the
spectra, zero where the Gaussian has no energy (above fmax).
"""
function reshape_source(v, dt, tr, w, delay; fmax=4.0)
    n = size(v, 1)
    L = nextpow(2, 2n + round(Int, 2delay / dt))
    f = (0:L÷2) ./ (L * dt)
    τ = ((0:L-1) .- (L ÷ 2)) .* dt
    W = SeismicWavePC.rfft(circshift(w(τ), -(L ÷ 2)))               # τ = 0 at the first sample
    G = SeismicWavePC.rfft([moment_rate(:gaussian, (k - 1) * dt, 0.0, tr) for k in 1:L])
    H = @. ifelse(f <= fmax, W * exp(-2π * im * f * delay) / G, 0.0)
    V = reshape(v, n, :)
    out = SeismicWavePC.irfft(H .* SeismicWavePC.rfft([V; zeros(L - n, size(V, 2))], 1), L, 1)[1:n, :]
    return reshape(out, size(v))
end
