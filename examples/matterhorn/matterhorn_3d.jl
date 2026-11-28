# 3D simulations of the Matterhorn (Limberger et al., 2025, Sci. Rep.; see model.jl): a vertically incident plane wave
# from below, as P wave and as S waves polarized north-south and east-west (one run each; the paper excites all three at
# once with x, y, z forces, which is their sum with the S waves 1.7× (vp/vs) stronger in velocity than the P wave).
# The incoming particle velocity is a Gaussian pulse (`stf = :gaussian`, rise time 0.5 s: energy up to ~4 Hz); the
# Ormsby (0.3-3.5 Hz) and Ricker (0.4 Hz) responses of the paper follow by spectral division (`reshape_source`).
# Stations: north-south and east-west profiles across the summit (5 km, 100 m apart) and a 250 m grid over 10 km × 10 km
# (the amplification map), all on the surface. Results are cached in examples/data/.
#
#   julia -t auto --project=examples/matterhorn examples/matterhorn/matterhorn_3d.jl [metal|cpu] [dx_km] [flat|ridges]
#
using SeismicWavePC, GeophysicalModelGenerator, Serialization, Printf
const DEVICE = length(ARGS) >= 1 ? ARGS[1] : "metal"
DEVICE == "metal" && using Metal
const backend = DEVICE == "metal" ? MetalBackend() : CPU()
const DX = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.075
const FLAT = length(ARGS) >= 3 && ARGS[3] == "flat"            # "ridges": the stations of Fig. S4 instead
include("model.jl")

const NA = 12                                                     # PML nodes
const TMAX, TRISE = 25.0, 0.5
const WAVES = ((:p, 0.0, "P"), (:sv, 180.0, "S_NS"), (:sv, 270.0, "S_EW"))   # (wave, backazimuth, name); SV at 180°: NS motion

"Stations: the NS and EW profiles (51 each) and the map grid (41 × 41)."
function matterhorn_stations()
    prof = [[Station(@sprintf("NS%+05d", round(Int, 1e3y)), 0.0, y, 0.0; surface=true) for y in -2.5:0.1:2.5];
            [Station(@sprintf("EW%+05d", round(Int, 1e3x)), x, 0.0, 0.0; surface=true) for x in -2.5:0.1:2.5]]
    grid = [Station(@sprintf("M%+05d%+05d", round(Int, 1e3x), round(Int, 1e3y)), x, y, 0.0; surface=true)
            for x in -5:0.25:5 for y in -5:0.25:5]
    return [prof; grid]
end

"""
Stations along the two ridge profiles of Fig. S4 of the paper (through the summit; read from its map): NW-SE along
x = -0.885 y (distance: the northing offset) and WSW-ENE along y = 0.39 x (the easting offset), 100 m apart, ±2.5 km.
"""
ridge_stations() = [[Station(@sprintf("RNS%+05d", round(Int, 1e3d)), -0.885d, d, 0.0; surface=true) for d in -2.5:0.1:2.5];
                    [Station(@sprintf("REW%+05d", round(Int, 1e3d)), d, 0.39d, 0.0; surface=true) for d in -2.5:0.1:2.5]]

function run3d(; dx=DX, flat=FLAT, stations=matterhorn_stations(), tag="")
    margin = (NA + SeismicWavePC.SBP_NB + 10) * dx
    L = BOX + margin
    n = round(Int, 2L / dx) + 1
    bottom = Z_BOT - (NA + SeismicWavePC.SBP_NB + 12) * 0.75dx
    mesh = matterhorn_mesh(range(-L, L, length=n), range(-L, L, length=n), matterhorn_surface(; flat); dz=0.75dx, bottom)
    model = matterhorn_model(mesh, dx)
    @printf("mesh %d × %d × %d = %.1f M nodes (dx = %.0f m, bottom %.2f km)\n", size(mesh)..., prod(size(mesh)) / 1e6, 1e3dx, bottom)
    for (wave, baz, name) in WAVES
        file = joinpath(@__DIR__, "..", "data", "matterhorn3d_$(tag)$(name)_$(round(Int, 1e3dx))m$(flat ? "_flat" : "").jls")
        isfile(file) && (println("have $file"); continue)
        w = TeleseismicWave(wave; angle=0.0, backazimuth=baz, box=(-BOX, BOX, -BOX, BOX, Z_BOT), t0=0.0, trise=TRISE, amplitude=1.0)
        time = @elapsed r = simulate(model; grid=:curvilinear, tmax=TMAX, sources=w, stations, stf=:gaussian, absorber=:pml, na=NA,
                                     backend, T=Float32, verbose=false)
        @printf("%-5s %5d steps, dt = %.2f ms: %.0f s\n", name, length(r.t), 1e3 * (r.t[2] - r.t[1]), time)
        mkpath(dirname(file))
        serialize(file, (; t=r.t, v=r.v, names=[s.name for s in stations], x=[s.x for s in stations], y=[s.y for s in stations],
                         dx, flat, time, nodes=prod(size(mesh))))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 3 && ARGS[3] == "ridges" ? run3d(; flat=false, stations=ridge_stations(), tag="ridges_") : run3d()
end
