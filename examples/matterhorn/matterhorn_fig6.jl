# Fig. 6 of Limberger et al. (2025): permafrost in the 2D north-south section of the Matterhorn (40 m grid, vertical P and
# SV plane waves, P + 1.7 SV, Ormsby source; see matterhorn_2d.jl). Models 1-5 of the paper (a permafrost cap with
# velocities × 1.3 and density × 1.1, 600, 450, 300, 150 m and 0 m thick below the summit: the paper shows the bodies only
# as a sketch, so the thicknesses of models 2-4 are our choice) and the sensitivity study (Fig. 6C): 30 models, thickness
# 100-600 m and velocity increase 10-50 %, amplitude reduction = 1 - PGV/PGV(model 5) of the horizontal motion at the
# summit.
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
