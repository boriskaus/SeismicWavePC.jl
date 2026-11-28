# Fig. 6 of Limberger et al. (2025): permafrost in the 2D north-south section of the Matterhorn (40 m grid, vertical P and
# SV plane waves, P + 1.7 SV, Ormsby source; see matterhorn_2d.jl). Models 1-5 of the paper (a permafrost cap with
# velocities × 1.3 and density × 1.1, 600, 450, 300, 150 m and 0 m thick below the summit: the paper shows the bodies only
# as a sketch, so the thicknesses of models 2-4 are our choice) and the sensitivity study (Fig. 6C): 30 models, thickness
# 100-600 m and velocity increase 10-50 %, amplitude reduction of the fundamental peak (0.25-0.6 Hz) of the horizontal
# summit spectrum relative to model 5.
#
#   julia -t 4 --project=examples/matterhorn examples/matterhorn/matterhorn_fig6.jl run k n   # worker k of n (k = 0 … n-1)
#   julia --project=examples/matterhorn examples/matterhorn/matterhorn_fig6.jl plot
#
include("matterhorn_2d.jl")

const MODELS15 = [(0.6, 1.3), (0.45, 1.3), (0.3, 1.3), (0.15, 1.3), (0.0, 1.0)]       # models 1-5
const THICK, FV = 0.1:0.1:0.6, 1.1:0.1:1.5                                           # Fig. 6C: 6 × 5 models
const ALL = unique([MODELS15; [(round(t, digits=2), round(f, digits=2)) for t in THICK for f in FV]])

if length(ARGS) >= 1 && ARGS[1] == "run"
    k, n = parse(Int, ARGS[2]), parse(Int, ARGS[3])
    for (q, (th, fv)) in enumerate(ALL)
        (q - 1) % n == k && run2d(th, fv)
    end
end

if length(ARGS) >= 1 && ARGS[1] == "plot"
    include("compare.jl")
    R = Dict(m => run2d(m...) for m in ALL)                      # (cached runs)
    r0 = R[(0.0, 1.0)]
    summit, valley = argmin(abs.(r0.x)), argmin(abs.(r0.x .+ 2.5))
    fg = collect(0:0.005:4.0)
    "Ormsby response (component c, station k) of a run: the trace and its smoothed amplitude spectrum on fg."
    function spec(r, c, k)
        dt = r.t[2] - r.t[1]
        o = vec(reshape_source(r.v[:, c:c, k:k], dt, 0.5, τ -> ormsby(τ), 4.0))
        f, S = spectrum(o, dt)
        j(q) = clamp(searchsortedlast(f, q), 1, length(f) - 1)
        return [S[j(q)] + (S[j(q)+1] - S[j(q)]) * (q - f[j(q)]) / (f[2] - f[1]) for q in fg], o
    end
    # amplitude reduction: of the fundamental resonance peak (0.25-0.6 Hz) of the horizontal summit spectrum, relative to
    # model 5 (as the peaks of the paper's Fig. 6B: model 1 -29 %). (The reduction of the summit PGV instead is not monotonic
    # in the thickness: interference in the time domain.)
    fpk(r) = maximum(spec(r, 1, summit)[1][0.25 .<= fg .<= 0.6])
    P5 = fpk(r0)
    red(m) = 100 * (1 - fpk(R[m]) / P5)                          # amplitude reduction [%]

    fig = Figure(size=(2000, 1300), fontsize=26, backgroundcolor=:white)
    # (A) the models: vp in the upper part of the section
    mesh, model, sec = section_model(DX2, NA2)
    X, Z = mesh.x.val[:, 1, :], mesh.z.val[:, 1, :]
    I = findall(x -> -1.4 <= x <= 1.5, X[:, end]); K = findfirst(k -> maximum(Z[I, k]) >= 2.0, axes(Z, 2)):size(Z, 2)
    nI, nK = length(I), length(K)
    verts = [vec(X[I, K]) vec(Z[I, K])]
    idx(i, k) = (k - 1) * nI + i
    faces = vcat([[idx(i, k) idx(i + 1, k) idx(i + 1, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...,
                 [[idx(i, k) idx(i + 1, k + 1) idx(i, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...)
    gA = fig[1, 1:2] = GridLayout()
    names = ["Model 1\nComplete permafrost", "Model 2\nHigh amount", "Model 3\nModerate amount", "Model 4\nSmall amount",
             "Model 5\nNo permafrost body"]
    local mA
    for (q, (th, fv)) in enumerate(MODELS15)
        m = th > 0 ? permafrost!(model, mesh, sec, th, fv)[1] : model
        ax = Axis(gA[1, q], aspect=DataAspect()); hidedecorations!(ax); hidespines!(ax)
        mA = mesh!(ax, verts, faces, color=1e3 .* vec(m.fields.vp[I, 1, K]), colormap=Reverse(:PuBuGn_9), colorrange=(3000, 5000),
                   shading=NoShading)
        text!(ax, 0.05, 2.8, text=names[q], align=(:center, :center), color=:white, fontsize=22, font=:italic)
        limits!(ax, -1.4, 1.5, 2.55, 4.45)
    end
    Colorbar(gA[1, 6], mA, label="Vp (m/s)", height=Relative(0.6), ticks=3000:500:5000)
    Label(gA[1, 1, TopLeft()], "(A)", fontsize=30, halign=:left)
    # (B) spectra at the summit (models 1-5) and in the valley
    gB = fig[2, 1] = GridLayout()
    axh = Axis(gB[1, 1], ylabel="Horizontal motion", title="(B)   Spectra at summit", titlealign=:left, titlesize=30, yticks=0.2:0.2:1.0)
    axv = Axis(gB[2, 1], ylabel="Vertical motion", xlabel="Frequency (Hz)")
    S5 = spec(r0, 1, summit)[1]; nrm = maximum(S5)
    labs = ["Complete permafrost body", "High amount", "Moderate amount", "Small amount", "No permafrost body"]
    for (m, lab, col) in zip(MODELS15, labs, (:black, :magenta, :red, :blue, :green))
        lines!(axh, fg, spec(R[m], 1, summit)[1] ./ nrm, color=col, label=lab)
        lines!(axv, fg, spec(R[m], 2, summit)[1] ./ nrm, color=col)
    end
    lines!(axh, fg, spec(r0, 1, valley)[1] ./ nrm, color=:black, linestyle=:dash, label="Valley reference")
    lines!(axv, fg, spec(r0, 2, valley)[1] ./ nrm, color=:black, linestyle=:dash)
    for ax in (axh, axv); xlims!(ax, 0, 4); end
    ylims!(axh, 0, 1.1); ylims!(axv, 0, 0.5); hidexdecorations!(axh, grid=false); rowgap!(gB, 0)
    rowsize!(gB, 2, Relative(0.4))
    axislegend(axh, position=:rt, framevisible=true, labelsize=18, rowgap=-4, padding=(6, 6, 2, 2))
    # (C) amplitude reduction over thickness and velocity increase (bilinear between the 6 × 5 models)
    th, fv = collect(THICK), collect(FV)
    Rd = [red((round(t, digits=2), round(f, digits=2))) for t in th, f in fv]
    tf, ff = range(th[1], th[end], length=101), range(fv[1], fv[end], length=81)
    function bil(t, f)
        i = clamp(searchsortedlast(th, t), 1, length(th) - 1); j = clamp(searchsortedlast(fv, f), 1, length(fv) - 1)
        a = (t - th[i]) / (th[i+1] - th[i]); b = (f - fv[j]) / (fv[j+1] - fv[j])
        return (1 - a) * (1 - b) * Rd[i, j] + a * (1 - b) * Rd[i+1, j] + (1 - a) * b * Rd[i, j+1] + a * b * Rd[i+1, j+1]
    end
    F = [bil(t, f) for t in tf, f in ff]
    axc = Axis(fig[2, 2], xlabel="Permafrost thickness below summit (m)", ylabel="Seismic velocity increase\ndue to permafrost (%)",
               title="(C)   Amplitude reduction sensitivity", titlealign=:left, titlesize=30)
    # (the paper's colour map, sampled from its colour bar: a cubehelix from dark purple over brown and olive to lavender)
    CH = cgrad([colorant"#2e1030", colorant"#45172c", colorant"#531b1f", colorant"#642212", colorant"#723104",
                colorant"#744500", colorant"#745603", colorant"#70671a", colorant"#6b763a", colorant"#65845c",
                colorant"#60917b", colorant"#649f9b", colorant"#7cacb7", colorant"#9bb9cf", colorant"#b8c5e5"])
    hm = heatmap!(axc, 1e3 .* tf, 100 .* (ff .- 1), F, colormap=CH, colorrange=(0, max(30, maximum(F))))
    contour!(axc, 1e3 .* tf, 100 .* (ff .- 1), F, levels=5:5:50, color=:black, linestyle=:dash, linewidth=1.5)
    Colorbar(fig[2, 3], hm, label="Amplitude reduction (%)")
    rowsize!(fig.layout, 1, Relative(0.3))
    ours = joinpath(@__DIR__, "matterhorn_fig6_40m.png")
    save(ours, fig)
    println("saved $ours")
    @printf("Fig. 6C (40 m): reduction of the fundamental peak (0.25-0.6 Hz) of the summit horizontal spectrum [%%], thickness (rows) × velocity +10…50 %% (columns)\n")
    for (i, t) in enumerate(th); @printf("  %4.0f m: %s\n", 1e3t, join((@sprintf("%5.1f", v) for v in Rd[i, :]), " ")); end
    @printf("models 1-4 (vs model 5): %s %%\n", join((@sprintf("%.1f", red(m)) for m in MODELS15[1:4]), ", "))
    paper = joinpath(PRIVATE, "paper_figs", "fig6.png")
    if isfile(paper)
        notes = wrap("Ours: 2D N-S section, curvilinear SBP-SAT, 40 m nodes, vertical P + 1.7 SV plane waves, Ormsby source " *
                     "(0.3-3.5 Hz); permafrost velocities × (1 + increase), density × 1.1, thickness below the summit thinning to 0 " *
                     "at 2.9 km (the paper shows the bodies only as a sketch: models 2-4 are 450, 300, 150 m, our choice). " *
                     "Amplitude reduction: of the fundamental peak (0.25-0.6 Hz) of the horizontal summit spectrum relative to model 5 (the " *
                     "paper does not define it; its Fig. 6B gives -29 % for model 1, ours -33 %). Spectra normalized to the " *
                     "maximum of model 5, smoothed over ±0.05 Hz.")
        file = compare_page(paper, ours, "compare_fig6.png"; top="Limberger et al. (2025), Fig. 6 (Salvus)",
                            bottom="SeismicWavePC, curvilinear SBP, 40 m grid", notes)
        println("saved $file")
    end
end
