# Shaded-relief map of the Matterhorn region from the raw swissALTIRegio DEM (10 m, as GMT's grdimage -I): elevation colours
# (green - yellow - brown - white) times a hillshade lit from the north-west, with longitude/latitude ticks (WGS84, from
# swisstopo's approximate LV95 -> WGS84 formulas, ~1 m) on a map in local LV95 coordinates [km, origin at the summit].
# Used by paper_style_3d.jl (Fig. 2B).
using CairoMakie, Printf
isdefined(@__MODULE__, :matterhorn_dem) || include("dem.jl")

"LV95 (E, N) [m] -> (longitude, latitude) [°], swisstopo's approximate formulas."
function lv95_to_wgs84(E, N)
    y, x = (E - 2_600_000) / 1e6, (N - 1_200_000) / 1e6
    λ = 2.6779094 + 4.728982y + 0.791484y * x + 0.1306y * x^2 - 0.0436y^3
    φ = 16.9023892 + 3.238272x - 0.270978y^2 - 0.002528x^2 - 0.0447y^2 * x - 0.0140x^3
    return λ * 100 / 36, φ * 100 / 36
end

const RELIEF_CMAP = cgrad([colorant"#3d7a1c", colorant"#9cc95a", colorant"#f3f0a8", colorant"#efd27a", colorant"#d9a441",
                           colorant"#a8692c", colorant"#7a4a24", colorant"#b9a493", colorant"#f2efec"],
                          [0.0, 0.12, 0.25, 0.4, 0.52, 0.68, 0.8, 0.9, 1.0])

"Hillshade (0-1) of heights `Z` [m] on a grid of spacing `h` [m], light from azimuth `az` and altitude `alt` [°]."
function hillshade(Z, h; az=315.0, alt=40.0, zfac=1.0)
    nx, ny = size(Z)
    S = similar(Z)
    a, e = deg2rad(az), deg2rad(alt)
    lx, ly, lz = cos(e) * sind(az), cos(e) * cosd(az), sin(e)            # light direction (x east, y north)
    for j in 1:ny, i in 1:nx
        zx = (Z[min(i + 1, nx), j] - Z[max(i - 1, 1), j]) / ((min(i + 1, nx) - max(i - 1, 1)) * h) * zfac
        zy = (Z[i, min(j + 1, ny)] - Z[i, max(j - 1, 1)]) / ((min(j + 1, ny) - max(j - 1, 1)) * h) * zfac
        S[i, j] = clamp((-zx * lx - zy * ly + lz) / sqrt(1 + zx^2 + zy^2), 0, 1)
    end
    return S
end

"""
Draw the relief map (|x|, |y| ≤ `half` km around the summit) into a new Axis at `pos`; returns the axis and the colour
range (for a Colorbar with `RELIEF_CMAP`).
"""
function relief_map!(pos; half=5.0, crange=(1700.0, 4300.0))
    d = matterhorn_dem()
    I = findall(x -> abs(x) <= half, d.x); J = findall(y -> abs(y) <= half, d.y)
    x, y, Z = d.x[I], d.y[J], 1e3 .* d.z[I, J]
    S = hillshade(Z, 10.0)
    col = [RELIEF_CMAP[clamp((Z[i, j] - crange[1]) / (crange[2] - crange[1]), 0, 1)] for i in axes(Z, 1), j in axes(Z, 2)]
    # GMT-like intensity: darken in shadow, lighten towards the light (0.5 neutral)
    function shade(c, s)
        k = clamp((s - 0.62) / 0.38, -1, 1)
        k < 0 ? RGBf(c.r * (1 + 0.5k), c.g * (1 + 0.5k), c.b * (1 + 0.5k)) :
                RGBf(c.r + (1 - c.r) * 0.3k, c.g + (1 - c.g) * 0.3k, c.b + (1 - c.b) * 0.3k)
    end
    img = [shade(col[i, j], S[i, j]) for i in axes(Z, 1), j in axes(Z, 2)]
    # longitude/latitude ticks where the meridians/parallels cross the bottom and left edges
    e0, n0 = d.summit[1], d.summit[2]
    geo(xk, yk) = lv95_to_wgs84(e0 + 1e3xk, n0 + 1e3yk)
    lons = [l for l in 7.0:1/30:8.0 if geo(-half, -half)[1] < l < geo(half, -half)[1]]    # every 2'
    lats = [l for l in 45.5:1/30:46.5 if geo(-half, -half)[2] < l < geo(-half, half)[2]]
    function bisect(f)                                     # (f increasing on [-half, half])
        a, b = -half, half
        for _ in 1:60
            m = (a + b) / 2
            f(m) > 0 ? (b = m) : (a = m)
        end
        return (a + b) / 2
    end
    xb(l) = bisect(xk -> geo(xk, -half)[1] - l)
    yl(l) = bisect(yk -> geo(-half, yk)[2] - l)
    dms(v, h) = (deg = floor(Int, v); mi = round(Int, 60(v - deg)); mi == 60 && (deg += 1; mi = 0); "$(deg)°$(lpad(mi, 2, '0'))'$h")
    ax = Axis(pos, aspect=DataAspect(), xticks=(xb.(lons), dms.(lons, "E")), yticks=(yl.(lats), dms.(lats, "N")),
              xgridvisible=false, ygridvisible=false)
    image!(ax, (x[1], x[end]), (y[1], y[end]), img)
    limits!(ax, -half, half, -half, half)
    text!(ax, -0.85half, -0.93half, text="N", fontsize=30, align=(:center, :bottom))
    arrows2d!(ax, [-0.85half], [-0.8half], [0.0], [0.13half], color=:black, shaftwidth=2, tipwidth=12, tiplength=12)
    return ax, crange
end
