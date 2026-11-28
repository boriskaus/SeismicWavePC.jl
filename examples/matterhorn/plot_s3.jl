# Fig. S3 of Limberger et al. (2025) from our 3D run (matterhorn_s3.jl): peak ground velocity (A) and peak deviatoric
# stress dQ at the surface, seen towards the south flank (B) and the north flank (C), each normalized to its maximum
# within 5 km of the summit, linear colour scales as the paper (PGV 0.1-1, dQ 0.01-1; its Paraview maps: batlowW-like
# for PGV, an ice-fire map for dQ, light cyan - blue - black - red - yellow, sampled by eye). With MATTERHORN_PRIVATE and
# the paper's figure in its paper_figs/figS3.png, also a comparison page (paper on top, ours below) there.
#
#   julia --project=examples/matterhorn examples/matterhorn/plot_s3.jl
#
using CairoMakie, Serialization, Printf
include("compare.jl")

r = deserialize(joinpath(@__DIR__, "..", "data", "matterhorn_s3_100m.jls"))
x, y, z = r.x ./ 1e3, r.y ./ 1e3, r.z ./ 1e3                     # km (x east, y north)
pgv, dq = r.pgv ./ maximum(r.pgv), r.dq ./ maximum(r.dq)
@printf("S3 (100 m): max PGV at (%.2f, %.2f) km; max dQ at (%.2f, %.2f) km, z = %.2f km\n",
        x[argmax(r.pgv)[1]], y[argmax(r.pgv)[2]], x[argmax(r.dq)[1]], y[argmax(r.dq)[2]], z[argmax(r.dq)])

const ICEFIRE = cgrad([colorant"#d8f6f8", colorant"#62c6e6", colorant"#1f5fb4", colorant"#0b1f4f", colorant"#000000",
                       colorant"#7a0b0b", colorant"#d23c0a", colorant"#f59a12", colorant"#fde74c"])
fig = Figure(size=(2400, 760), fontsize=28, backgroundcolor=:white)
panels = (("(A) Amplification map", pgv, :batlowW, (0.1, 1.0), "PGV", 1.25π),     # (camera azimuths: 1.5π from the south)
          ("(B) Stress (south flank)", dq, ICEFIRE, (0.01, 1.0), "dQ", 1.5π),
          ("(C) Stress (north flank)", dq, ICEFIRE, (0.01, 1.0), "dQ", 0.5π))
for (j, (title, F, cmap, cr, lab, az)) in enumerate(panels)
    Label(fig[1, j], title, fontsize=34, halign=:left, tellwidth=false)
    ax = Axis3(fig[2, j]; azimuth=az, elevation=0.12π, aspect=:data, perspectiveness=0.4, protrusions=0, viewmode=:stretch)
    hidedecorations!(ax); hidespines!(ax)
    s = surface!(ax, x, y, z; color=clamp.(F, cr...), colormap=cmap, colorrange=cr, shading=NoShading)
    j != 2 && Colorbar(fig[2, j], s; label=lab, width=20, height=200, halign=:right, valign=:bottom, tellwidth=false,
                       tellheight=false, ticks=[cr[1], cr[2]])
end
rowsize!(fig.layout, 2, Relative(0.9))
ours = joinpath(@__DIR__, "matterhorn_figS3_100m.png")
save(ours, fig)
println("saved $ours")
paper = joinpath(PRIVATE, "paper_figs", "figS3.png")
if isfile(paper)
    notes = wrap("Colour scales linear (as Fig. 7 of the paper; the paper's maps sampled by eye). Ours: 3D, curvilinear SBP-SAT, 100 m nodes (3.8 M), Float32 on the GPU, vertical plane waves (P + 1.7 SV, both " *
                 "S polarizations as matterhorn_3d.jl), 0.4 Hz Ricker as in the paper's Fig. 5A; peaks over 25 s, each normalized " *
                 "to its maximum within 5 km of the summit. dQ = √J2 from the strain of the integrated surface displacement. " *
                 "Paper: Salvus (elements < 200 m), shown in Paraview.")
    file = compare_page(paper, ours, "compare_figS3.png"; top="Limberger et al. (2025), Fig. S3 (Salvus)",
                        bottom="SeismicWavePC, curvilinear SBP, 100 m grid", notes)
    println("saved $file")
end
