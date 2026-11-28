# Matterhorn reproduction: all simulations

Every simulation behind the figures of this example (reproduction of Limberger et al., 2025, Sci. Rep. 15, 23914).
Solver: SeismicWavePC.jl, curvilinear grid (`grid=:curvilinear`): collocated velocity-stress, boundary-optimized upwind
SBP operators of order 7 (Mattsson-Niemelä-Winters), SAT free surface and interfaces, leapfrog time stepping with the time
step from a power iteration, CFS-PML absorbing layers (`absorber=:pml`), attenuation with 3 relaxation mechanisms
(Qp = 200, Qs = 100, constant over 0.05-5 Hz; elastic within 6 nodes of the injection box), plane waves by
total-field/scattered-field injection along a box (`TeleseismicWave`; exact incoming field of the horizontally layered
background, one layer per mesh row). Machine: Apple MacBook Pro, M4 Max (64 GB); GPU runs with Metal in Float32, CPU runs
with 4 threads in Float64. Wall times are for runs that shared the machine with other jobs (± 30 %).

Common model (model.jl): DEM swissALTIRegio (swisstopo, 10 m), Gaussian-smoothed with σ, real within 5 km of the summit
and blended to a flat surface at 2.95 km between 5 and 6.5 km; vp 3 → 6 km/s linearly from the summit (4.478 km) to
-2 km (constant below), vs = vp/1.7, density 2600 → 3000 kg/m³ in the same way; injection box |x|, |y| ≤ 7 km down to
-2 km; mesh rows flat below -1.5 km and following the topography above (C2 blend); vertical node spacing 0.75 dx.
Incoming waves: P, SV polarized N-S and SV polarized E-W (3D) or P and SV (2D), each as its own run with a Gaussian
moment-rate (rise 0.5 s, 3D 1 s); the paper's source wavelets (Ormsby 0.3-3.5 Hz, Ricker 0.4 Hz) are applied afterwards
by spectral division (`reshape_source`), or given directly as a sampled source (Fig. 7, S3). Sum: P + 1.7 (S waves), as
equal forces radiate them (vp/vs = 1.7).

## 3D runs (matterhorn_3d.jl, analysis analyze_3d.jl, paper_style_3d.jl, plot_s4.jl)

| purpose | dx | nodes | dt | steps (25 s) | backend | wall time per incoming wave | σ (DEM) | stations |
|---|---|---|---|---|---|---|---|---|
| Figs. 2, 4A-C, 5A (main) | 60 m | 11.9 M (290 × 290 × 141) | 1.294 ms | 19 318 | Metal, Float32 | ~15 min | 40 m | 1783 (2 lines, 41 × 41 map) |
| convergence check | 100 m | 3.8 M | 2.578 ms | 9 696 | Metal, Float32 | ~3 min | 40 m | 1783 |
| control: flat surface | 100 m | 3.8 M | 3.791 ms | 6 594 | Metal, Float32 | ~2 min | flat | 1783 |
| control: flat surface | 200 m | 1.0 M | 8.603 ms | 2 906 | Metal, Float32 | < 1 min | flat | 1783 |
| Fig. S4 (ridge profiles) | 60 m | 11.9 M | 1.294 ms | 19 318 | Metal, Float32 | 1032 / 966 / 951 s (P / SV_NS / SV_EW), peak memory 12 GB | 40 m | 102 (2 ridge lines) |
| Fig. S3 (surface PGV and dQ, 0.4 Hz Ricker) | 100 m | 3.8 M | 2.58 ms | 9 696 | Metal, Float32 | 280 / 334 / 442 s (the last with the stress callback), peak memory 7.4 GB | 40 m | surface nodes within 5 km, every 10 steps |

Flat-surface control: all summit/valley ratios 1.00, the plane wave uniform to 0.5-2 %. Convergence: 100 m vs 60 m
reproduce the summit resonance frequencies (0.42 Hz NS, 0.46 Hz EW at 60 m).

## 2D runs, north-south section through the summit (matterhorn_2d.jl, matterhorn_fig6.jl, matterhorn_fig7.jl)

Section: 2D P-SV, x north, |x| ≤ 7 km + absorbing margins, 20 PML nodes.

| purpose | dx | nodes | dt | steps | backend | wall time | σ (DEM) | notes |
|---|---|---|---|---|---|---|---|---|
| Fig. 6: 33 permafrost models (models 1-5 and 6 × 5 sensitivity grid) | 40 m | 86 292 (423 × 204) | 1.019 ms | 24 532 (25 s) | CPU, 4 threads, Float64 | 186-364 s per model (P + SV; median 251 s) | 40 m | permafrost: velocities × (1 + increase), density × 1.1, thickness below the summit thinning to 0 at 2.9 km |
| Fig. 7 main: 0°, and 45° at z = 0 from N and S (56.7° in the half-space), both P/SV polarities | 40 m | 86 292 | 1.394 ms | 17 934 (25 s) | Metal, Float32 | ~30 s per SV run (+ P run), peak memory 8.8 GB | 100 m | Ormsby source sampled; per-node peaks of |v_h|, |v_z|, dQ every 7 steps (2563 snapshots) |
| Fig. 7 convergence (45° in the half-space) | 30 m | — | 1.044 ms | 23 946 | Metal, Float32 | ~14 min per case | 100 m | surface dQ maxima within 1-3 kPa (≤ 10 %) of 40 m |
| Fig. 7 grid check | 60 m | — | 2.120 ms | 11 793 | CPU, Float64 | ~5-6 min per case | 100 m | |
| Fig. 7 with σ = 40 m (not converged) | 60 / 40 m | — | 1.541 / 1.019 ms | — | CPU / Metal | ~15 / 10 min per case | 40 m | north-side corner stress differs up to 2× between 60 and 40 m: reason for σ = 100 m |

## Choices where the paper gives no value

| quantity | ours | reason |
|---|---|---|
| density | 2600-3000 kg/m³, linear with vp | read from the colour bar of the paper's Fig. 1 |
| relative amplitude and polarity of P and S | P + 1.7 S, both polarities tested for Fig. 7 (flank stress ±10-15 %) | equal forces; polarity not stated |
| angle of the oblique waves (Fig. 7) | 45° at z = 0 (56.7° below -2 km) | the paper tilts its source plane by 45° at the surface |
| time origin, source depth | Figs. 2/4/S4 time axis shifted by -1.2 s (valley EW peak at 4.4 s as in the paper) | not given |
| permafrost geometry (Fig. 6) | cap of given thickness below the summit, thinning to 0 at 2.9 km; models 2-4: 450, 300, 150 m | shown only as a sketch |
| amplitude reduction (Fig. 6C) | reduction of the fundamental peak (0.25-0.6 Hz) of the horizontal summit spectrum | not defined; matches its Fig. 6B (model 1: -29 % there, -33 % here) |
| DEM smoothing | σ = 40 m (3D, Fig. 6), 100 m (Fig. 7) | paper: "slightly smoothed"; σ = 100 m needed for grid-independent stresses |
| stress measure | dQ = √J2 of the deviatoric stress, plane strain σyy = ν(σxx + σzz) (2D) | the paper's supplement |

## Memory notes

The first Fig. 6/7 attempts (2026-11-28) exhausted the 64 GB laptop (kernel watchdog panic): the TeleseismicWave tables
needed tens of GB for long runs, fixed in SeismicWavePC (PR #15). All runs since go through `run_guarded.sh` (kills a job
above its memory limit or when the machine drops below 15 % free memory).
