# Push-Optimal Sokoban Solver

A push-optimal Sokoban solver built for make.fun's
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty.

## How it works

- Solves each room independently from its door: a level's push-optimal
  count is the sum of its rooms' optima, and a box pushed into the
  hallway can never come back, so rooms never interact.
- Push-based A* with a min-assignment heuristic over single-box push
  distances, deadlock pruning (corners, 2x2 blocks, frozen boxes), and
  a deterministic node budget — no clock, no randomness, so both runs
  produce byte-identical output.

## Run

`node main.ts` reads `/input/<id>/level.txt` for i001–i005 and writes
`/output/<id>/solution.txt` (LURD move strings). Node 22, no dependencies.
