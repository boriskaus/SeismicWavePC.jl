# Figs. 1A and S2 of Limberger et al. (2025) for our models: the 3D Matterhorn block (the smoothed DEM of the 3D runs,
# σ = 40 m, coloured by vp, 10 × 10 km around the summit, down to the bottom of the velocity gradient at -2 km) with the
# incoming plane wave, and the 2D north-south section with its curvilinear mesh (40 m, as the Fig. 6 runs; every second
# mesh line drawn). With MATTERHORN_PRIVATE and the paper's figures in its paper_figs/, comparison pages there.
#
#   julia --project=examples/matterhorn examples/matterhorn/plot_models.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf
include("model.jl")
include("compare.jl")
# the paper's vp colour map (Paraview "viridis"-like, from white-beige to navy), sampled from its Fig. S2 colour bar
const VPMAP = cgrad([colorant"#fffaf2", colorant"#ebe298", colorant"#afd16f", colorant"#4eaf5e", colorant"#34997d",
                     colorant"#237c8b", colorant"#1e5285", colorant"#24367f", colorant"#2b2a66"])

# --- Fig. 1A: the 3D block
topo = matterhorn_surface()
xs = range(-5, 5, length=201)
fig = Figure(size=(1500, 950), fontsize=24, backgroundcolor=:white)
Label(fig[1, 1], "(A) Matterhorn (Switzerland)", fontsize=34, halign=:left, tellwidth=false)
ax = Axis3(fig[2, 1], azimuth=1.62π, elevation=0.12π, aspect=(1, 1, 0.65), perspectiveness=0.25, xlabel="east (km)",
           ylabel="north (km)", zlabel="Elevation (km)", protrusions=(60, 20, 20, 20))
T = [topo(a, b) for a in xs, b in xs]
cr = (3.0, 6.0)
surface!(ax, xs, xs, T; color=vp_of.(T), colormap=VPMAP, colorrange=cr, shading=NoShading)
zb = Z_BOT
for (side, pts) in (("south", [(a, -5.0) for a in xs]), ("east", [(5.0, b) for b in xs]))
    zz = range(0, 1, length=60)
    X = [p[1] for p in pts, _ in zz]; Y = [p[2] for p in pts, _ in zz]
    Z = [zb + s * (topo(p...) - zb) for p in pts, s in zz]
    surface!(ax, X, Y, Z; color=vp_of.(Z), colormap=VPMAP, colorrange=cr, shading=NoShading)
end
xa = collect(-4.6:0.5:4.6)
arrows3d!(ax, Point3f.(xa, -5.0, zb + 0.1), Vec3f.(0, 0, 0.9), color=:white, shaftradius=0.02, tipradius=0.06, tiplength=0.2)
text!(ax, Point3f(1.0, -5.0, zb + 1.3), text="Incoming plane wave", color=:white, fontsize=28)
text!(ax, Point3f(0.0, 0.0, topo(0.0, 0.0) + 0.35), text="Matterhorn Summit\n$(round(Int, 1e3topo(0.0, 0.0))) m", align=(:center, :bottom),
      fontsize=22, font=:bold)
text!(ax, Point3f(0.0, -2.5, topo(0.0, -2.5) + 0.35), text="Valley\n$(round(Int, 1e3topo(0.0, -2.5))) m", align=(:center, :bottom),
      fontsize=22, font=:bold)
for (a, b) in ((0.0, 0.0), (0.0, -2.5))                         # (leader lines from the labels)
    lines!(ax, [a, a], [b, b], [topo(a, b), topo(a, b) + 0.32], color=:black, linewidth=3)
end
zlims!(ax, zb, 5.3)
Colorbar(fig[2, 2], colormap=VPMAP, colorrange=(3000, 6000), label="Vp (m/s); ρ 2600-3000 kg/m³; Vs = Vp/1.7", height=Relative(0.5))
ours1 = joinpath(@__DIR__, "matterhorn_fig1a.png"); save(ours1, fig); println("saved $ours1")

# --- Fig. S2: the 2D section and its mesh (x: 6000 m at the summit, as the paper's axes)
mesh, model, sec = section_model(0.04, 20)
X, Z = 1e3 .* mesh.x.val[:, 1, :] .+ 6000, 1e3 .* mesh.z.val[:, 1, :]
vp = 1e3 .* model.fields.vp[:, 1, :]
I = findall(x -> 2000 <= x <= 9800, X[:, end]); K = findfirst(k -> minimum(Z[I, k]) >= -2000, axes(Z, 2)) - 1:size(Z, 2)
nI, nK = length(I), length(K)
verts = [vec(X[I, K]) vec(Z[I, K])]
idx(i, k) = (k - 1) * nI + i
faces = vcat([[idx(i, k) idx(i + 1, k) idx(i + 1, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...,
             [[idx(i, k) idx(i + 1, k + 1) idx(i, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...)
fig = Figure(size=(1500, 1000), fontsize=24, backgroundcolor=:white)
ax = Axis(fig[1, 1], xlabel="x (m), south → north", ylabel="Elevation (m)", aspect=DataAspect())
m = mesh!(ax, verts, faces, color=vec(vp[I, K]), colormap=VPMAP, colorrange=(3000, 5000), shading=NoShading)
for i in I[1:2:end]; lines!(ax, X[i, K], Z[i, K], color=(:white, 0.6), linewidth=0.5); end
for k in K[1:2:end]; lines!(ax, X[I, k], Z[I, k], color=(:white, 0.6), linewidth=0.5); end
limits!(ax, 2000, 9800, -2000, 4500)
Colorbar(fig[1, 2], m, label="VP (m/s)", height=Relative(0.4))
ours2 = joinpath(@__DIR__, "matterhorn_figS2.png"); save(ours2, fig); println("saved $ours2")
@printf("S2: %d × %d nodes (40 m), summit of the section %.0f m\n", size(X)..., maximum(Z))

for (paper, ours, name, notes, crop) in
    ((paper_fig(1), ours1, "compare_fig1.png", "Paper cropped to the Matterhorn panel (A). Ours: the model of the 3D runs, the DEM smoothed " *
      "with σ = 40 m, real within 5 km of the summit (blended to a flat surface at 2.95 km between 5 and 6.5 km, outside the " *
      "plot), vp 3 → 6 km/s from the summit to -2 km (constant below), density 2600-3000 kg/m³ (read from the paper's Fig. 1), " *
      "vs = vp/1.7, Qp/Qs = 200/100; the plane wave is injected along a box (total-field/scattered-field) whose bottom is at -2 km.",
      (:, 1:1020)),
     (joinpath(PRIVATE, "paper_figs", "figS2.png"), ours2, "compare_figS2.png", "Ours: the 2D section of Figs. 6 and 7, curvilinear " *
      "mesh with 40 m node spacing (every second line drawn; rows flat below -1.5 km, following the topography above), " *
      "DEM smoothed with σ = 40 m (Fig. 6; σ = 100 m for Fig. 7). The paper's mesh lines are ≈ 60 m apart.", nothing))
    isfile(paper) && println("saved ", compare_page(paper, ours, name; top="Limberger et al. (2025), $(replace(name, "compare_fig" => "Fig. ", ".png" => ""))",
                                                    bottom="SeismicWavePC model", notes=wrap(notes), crop))
end
