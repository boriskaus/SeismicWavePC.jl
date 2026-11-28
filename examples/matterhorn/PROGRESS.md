# Matterhorn reproduction: progress log (for resuming after an interruption)

Branch `matterhorn` (worktree `.claude/worktrees/agent-ab777207b4b90ffc2`), not to be merged yet. Results cache:
`examples/data/` (git-ignored, survives reboots). Never keep anything needed later in `/private/tmp` (wiped at boot).

## Memory rule (the 2026-11-28 15:28 reboot)
Kernel watchdog panic from memory exhaustion: 4 Julia processes at once (Fig. 7 Metal run 51 GB, three Fig. 6 sweep
workers 39 GB each) on the 64 GB laptop. Run heavy jobs one at a time (at most one GPU + one CPU job), always with
`--heap-size-hint=8G`, and under `run_guarded.sh`, which kills a job above its RSS limit.

## Done
- 3D runs 60 m / 100 m / flat controls; 2D section spectra and permafrost (models 1, 5, +50 %); Fig. 7 at 60/40/30 m with
  the angle 45° in the half-space (converged at 40 m with σ = 100 m DEM smoothing). README has the numbers.
- Fig. 7 paper-style plot with equal aspect, the paper's colour maps (batlowW, vik), stacked comparison helper
  (`compare.jl`), resolution footnote.
- Report methods section (LaTeX) in `report/` (main.tex, methods.tex).

## To do (in this order)
1. Fig. 7, 45° at z = 0 (56.7° in the half-space), both P/S polarities from the same runs:
   `matterhorn_fig7.jl 0.04 metal z0 plus` -> `matterhorn_fig7_*_40m_s100_pm.jls`; then `fig7_paper_style.jl 0.04
   ../data/paper_figs/fig7.png z0 plus main` and `... minus opposite` (choose the polarity closer to the paper).
2. Fig. 6 sweep (35 2D models): `matterhorn_fig6.jl run 0 1` (ONE worker), then `matterhorn_fig6.jl plot`.
3. Fig. S3 surface PGV/dQ map: `matterhorn_s3.jl metal` (3D 100 m).
4. Fig. S4 ridge profiles: `matterhorn_3d.jl metal 0.06 ridges`.
5. Stacked comparisons (paper on top, ours below, A4 width) for Figs. 1A, 2, 4A-C, 5A, 6, 7, S2-S4; `RUNS.md` with
   every run (grid, nodes, dt, steps, backend, precision, wall time, smoothing, convergence).
6. Report: results sections per figure, run table, unspecified inputs (P/S polarity, density, permafrost geometry),
   reproduction instructions. Open question for the user: authors/affiliation on the title page.

Private material (paper figures, comparison pages, the report with the paper's figures) lives OUTSIDE the repository
in `~/Documents/Matterhorn_private` (a local git repository; run figure scripts with
`MATTERHORN_PRIVATE=~/Documents/Matterhorn_private`). The branch holds only our scripts and model results. Paper figures
(CC BY 4.0) from the (Springer media server
`https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-08218-5/MediaObjects/41598_2025_8218_Fig{n}_HTML.png`).

## Log
- 2026-11-28 15:40: resumed after the reboot; WIP scripts committed (61548f1), report draft recovered (94c8750).
- 2026-11-28 15:41: started task 1 (Fig. 7 z0, Metal, guard 16 GB: footprint ~6 GB) and task 2 (Fig. 6 sweep, one CPU
  worker, guard 12 GB). Logs and footprint traces in examples/data/logs/. Paper figures re-downloaded to
  examples/data/paper_figs/ (supplement not yet).
- 2026-11-28 15:43-16:20: the guard killed the restarted jobs (setup spikes of 23-57 GB). Cause: TeleseismicWave injection
  tables (full spectra per depth, per-column irfft garbage, depth grid at 1/8 of the SBP surface spacing). Fixed in
  114f544 + a550403 (setup now ~3 GB; seismograms change ≤ 5e-6); PR #15 to main (fix-injection-memory).
- 2026-11-28 16:28: tasks 1 and 2 restarted with the fix (footprints 2.2 and 5.2 GB). Material from the paper is no longer
  tracked: comparison pages go to examples/data/comparisons, report PDF ignored (the old composite is still in the
  history, commit f94ad8b).
- 2026-11-28 16:55: private folder ~/Documents/Matterhorn_private (paper figures, comparison pages, report sources, local
  git); report removed from the branch.
