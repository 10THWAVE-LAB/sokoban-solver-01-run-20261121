// Sokoban solver: push-optimal per room, then concatenated.
//
// Structure of every level in this competition (see the kit rules):
// a 1-wide hallway holds the player, with 1-4 rooms off it, each room
// opening through exactly one door, each box staying in its room, and
// a level's push-optimal count being the sum of its rooms' optima.
// So: split the level at the hallway row, solve each room on its own
// with push-based A* (admissible min-assignment heuristic over
// precomputed single-box push distances, plus deadlock pruning), and
// join the rooms' move strings with hallway walks between doors.
// Determinism is by construction: fixed iteration orders, a
// deterministic heap tie-break, and a budget counted in node
// expansions only -- no clock, no randomness, no environment reads.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const INF = 1000000000;
// Directions in a fixed order: up, down, left, right. opp = dir ^ 1.
const DR = [-1, 1, 0, 0];
const DC = [0, 0, -1, 1];
const CH = ['u', 'd', 'l', 'r'];

// Whole-run work budget (node expansions), sized far above what the
// practice rooms need (~15-60 each) and far below the 60 s wall.
const BUDGET = { used: 0, limit: 600000 };
const ROOM_ASTAR_CAP = 150000;
const FALLBACK_CAP = 150000;

const keyOf = (r: number, c: number): number => r * 4096 + c;

interface Level {
  h: number;
  w: number;
  walls: Set<number>;
  goals: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.replace(/\r/g, '').split('\n').filter((l) => l.length > 0);
  const h = rows.length;
  let w = 0;
  for (const row of rows) w = Math.max(w, row.length);
  const walls = new Set<number>();
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = -1;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const ch = c < rows[r].length ? rows[r][c] : '#';
      const k = keyOf(r, c);
      if (ch === '#') walls.add(k);
      else if (ch === '.') goals.add(k);
      else if (ch === '$') boxes.push(k);
      else if (ch === '*') { boxes.push(k); goals.add(k); }
      else if (ch === '@') player = k;
      else if (ch === '+') { player = k; goals.add(k); }
    }
  }
  boxes.sort((a, b) => a - b);
  return { h, w, walls, goals, boxes, player };
}

// ------------------------------------------------------------------
// Subproblem: a set of floor cells (room cells + optional entry cell),
// boxes that may only occupy the room cells, goals inside the room.
// Cell indices are assigned in (row, col) order, so every iteration
// order and every "minimum index" region representative is fixed.
// ------------------------------------------------------------------
interface Sub {
  n: number;
  rc: Array<[number, number]>;
  idx: Map<number, number>;
  nb: number[][]; // nb[i][dir] = neighbour index, or -1 for wall/outside
  room: boolean[]; // a box may stand here
  goals: number[];
  goalSet: Set<number>;
  startBoxes: number[];
  startPlayer: number;
  door: number; // index of the door cell, or -1
  entry: number; // index of the hallway cell outside the door, or -1
  requireExit: boolean;
  dist: Array<Map<number, number>>; // dist[cell] : goal -> min pushes (single box)
  heurCache: Map<string, number>;
}

function buildSub(
  cells: Array<[number, number]>,
  roomKeys: Set<number>,
  goals: number[],
  boxes: number[],
  player: number,
  doorKey: number,
  entryKey: number,
  requireExit: boolean,
): Sub {
  const sorted = cells.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const idx = new Map<number, number>();
  sorted.forEach((cell, i) => idx.set(keyOf(cell[0], cell[1]), i));
  const n = sorted.length;
  const nb: number[][] = [];
  for (let i = 0; i < n; i++) {
    const [r, c] = sorted[i];
    const row: number[] = [];
    for (let d = 0; d < 4; d++) {
      const k = idx.get(keyOf(r + DR[d], c + DC[d]));
      row.push(k === undefined ? -1 : k);
    }
    nb.push(row);
  }
  const room = sorted.map((cell) => roomKeys.has(keyOf(cell[0], cell[1])));
  const goalIdx = goals.map((k) => idx.get(k)).filter((v) => v !== undefined) as number[];
  const startBoxes = boxes.map((k) => idx.get(k)).filter((v) => v !== undefined) as number[];
  startBoxes.sort((a, b) => a - b);
  const sub: Sub = {
    n, rc: sorted, idx, nb, room,
    goals: goalIdx,
    goalSet: new Set(goalIdx),
    startBoxes,
    startPlayer: idx.get(player) ?? -1,
    door: doorKey >= 0 ? (idx.get(doorKey) ?? -1) : -1,
    entry: entryKey >= 0 ? (idx.get(entryKey) ?? -1) : -1,
    requireExit,
    dist: [],
    heurCache: new Map(),
  };
  sub.dist = buildDist(sub);
  return sub;
}

// Flood fill from `start` over cells not in `blocked`; returns the
// region's minimum cell index (the canonical player representative)
// and the region itself.
function region(sub: Sub, blocked: Set<number>, start: number): { rep: number; set: Set<number> } {
  const set = new Set<number>();
  if (start < 0 || blocked.has(start)) return { rep: -1, set };
  const stack = [start];
  set.add(start);
  let rep = start;
  while (stack.length > 0) {
    const cur = stack.pop() as number;
    if (cur < rep) rep = cur;
    for (let d = 0; d < 4; d++) {
      const nxt = sub.nb[cur][d];
      if (nxt >= 0 && !blocked.has(nxt) && !set.has(nxt)) {
        set.add(nxt);
        stack.push(nxt);
      }
    }
  }
  return { rep, set };
}

// Single-box push distances, ignoring other boxes: for every room
// cell, a BFS over (box, player-region) to each goal. This is a lower
// bound on the true push distance, hence admissible as a heuristic.
function buildDist(sub: Sub): Array<Map<number, number>> {
  const dist: Array<Map<number, number>> = [];
  for (let i = 0; i < sub.n; i++) dist.push(new Map());
  for (let s = 0; s < sub.n; s++) {
    if (!sub.room[s]) continue;
    const res = dist[s];
    const seen = new Set<number>();
    const q: Array<[number, number, number]> = [];
    const reps = new Set<number>();
    for (let c = 0; c < sub.n; c++) {
      if (c === s) continue;
      reps.add(region(sub, new Set([s]), c).rep);
    }
    for (const rep of reps) {
      if (rep < 0) continue;
      const k = s * 4096 + rep;
      if (!seen.has(k)) { seen.add(k); q.push([s, rep, 0]); }
    }
    let head = 0;
    while (head < q.length) {
      const [box, prep, d] = q[head++];
      if (sub.goalSet.has(box) && !res.has(box)) {
        res.set(box, d);
        if (res.size === sub.goals.length) break;
      }
      const reg = region(sub, new Set([box]), prep);
      for (let dir = 0; dir < 4; dir++) {
        const dest = sub.nb[box][dir];
        if (dest < 0) continue;
        const behind = sub.nb[box][dir ^ 1];
        if (behind < 0 || !reg.set.has(behind)) continue;
        const nrep = region(sub, new Set([dest]), box).rep;
        const k = dest * 4096 + nrep;
        if (!seen.has(k)) { seen.add(k); q.push([dest, nrep, d + 1]); }
      }
    }
  }
  return dist;
}

// Minimum-assignment heuristic: the cheapest way to match boxes to
// distinct goals using single-box push distances. Each push moves one
// box, so the sum of per-box minima (over a matching) lower-bounds
// the pushes still needed. Falls back to a sum of per-box minima
// when there are too many boxes to permute.
function heuristic(sub: Sub, boxes: number[]): number {
  const cacheKey = boxes.join(',');
  const cached = sub.heurCache.get(cacheKey);
  if (cached !== undefined) return cached;
  let best = INF;
  if (boxes.length <= 5) {
    const goals = sub.goals;
    const used = new Array<boolean>(goals.length).fill(false);
    const perm: number[] = [];
    const rec = (i: number, sum: number): void => {
      if (sum >= best) return;
      if (i === boxes.length) { best = sum; return; }
      for (let g = 0; g < goals.length; g++) {
        if (used[g]) continue;
        const d = sub.dist[boxes[i]].get(goals[g]);
        if (d === undefined) continue;
        used[g] = true;
        perm.push(g);
        rec(i + 1, sum + d);
        perm.pop();
        used[g] = false;
      }
    };
    rec(0, 0);
  } else {
    let sum = 0;
    for (const b of boxes) {
      let m = INF;
      for (const g of sub.goals) {
        const d = sub.dist[b].get(g);
        if (d !== undefined && d < m) m = d;
      }
      if (m === INF) { sum = INF; break; }
      sum += m;
    }
    best = sum;
  }
  sub.heurCache.set(cacheKey, best);
  return best;
}

// Deadlock tests. A box off a goal is dead when: it can reach no goal
// at all (static dead cell); it sits in a corner; it is frozen against
// a wall (wall on one side, blocked on both perpendicular sides); or
// it is part of a 2x2 block of boxes/walls with no goal in the block.
function isDead(sub: Sub, boxes: number[], boxSet: Set<number>): boolean {
  for (const b of boxes) {
    if (sub.goalSet.has(b)) continue;
    if (sub.dist[b].size === 0) return true;
    const up = sub.nb[b][0]; const dn = sub.nb[b][1];
    const lf = sub.nb[b][2]; const rt = sub.nb[b][3];
    if ((up < 0 && lf < 0) || (up < 0 && rt < 0) || (dn < 0 && lf < 0) || (dn < 0 && rt < 0)) return true;
    const sides: Array<[number, number, number]> = [
      [up, lf, rt], [dn, lf, rt], [lf, up, dn], [rt, up, dn],
    ];
    for (const [wall, p1, p2] of sides) {
      if (wall < 0) {
        const b1 = p1 < 0 || boxSet.has(p1);
        const b2 = p2 < 0 || boxSet.has(p2);
        if (b1 && b2) return true;
      }
    }
  }
  for (const b of boxes) {
    const [r, c] = sub.rc[b];
    for (const [dr, dc] of [[0, 0], [0, -1], [-1, 0], [-1, -1]] as const) {
      const sq: Array<[number, number]> = [
        [r + dr, c + dc], [r + dr, c + dc + 1],
        [r + dr + 1, c + dc], [r + dr + 1, c + dc + 1],
      ];
      let allBlocked = true;
      let hasOffGoalBox = false;
      for (const [sr, sc] of sq) {
        const si = sub.idx.get(keyOf(sr, sc));
        if (si === undefined || boxSet.has(si)) {
          if (si !== undefined && boxSet.has(si) && !sub.goalSet.has(si)) hasOffGoalBox = true;
        } else {
          allBlocked = false;
        }
      }
      if (allBlocked && hasOffGoalBox) return true;
    }
  }
  return false;
}

// Player walk (no pushes) from `from` to `to` over cells not in
// `blocked`, as a move string; null when unreachable.
function walkPath(sub: Sub, blocked: Set<number>, from: number, to: number): string | null {
  if (from === to) return '';
  if (from < 0 || to < 0) return null;
  const prev = new Map<number, [number, number]>();
  const q = [from];
  prev.set(from, [-1, -1]);
  let head = 0;
  while (head < q.length) {
    const cur = q[head++];
    for (let d = 0; d < 4; d++) {
      const nxt = sub.nb[cur][d];
      if (nxt < 0 || blocked.has(nxt) || prev.has(nxt)) continue;
      prev.set(nxt, [cur, d]);
      if (nxt === to) {
        let s = '';
        let node = to;
        while (node !== from) {
          const [p, dir] = prev.get(node) as [number, number];
          s = CH[dir] + s;
          node = p;
        }
        return s;
      }
      q.push(nxt);
    }
  }
  return null;
}

interface PushStep { boxFrom: number; boxTo: number; dir: number; }
interface ParentInfo { parent: string; step: PushStep; }
interface SubResult { pushes: number; moves: string; finalPlayer: number; }

function stateKey(boxes: number[], prep: number): string {
  return boxes.join(',') + '|' + prep;
}

function isGoalState(sub: Sub, boxes: number[]): boolean {
  for (const b of boxes) if (!sub.goalSet.has(b)) return false;
  return true;
}

function exitReachable(sub: Sub, boxes: number[], player: number): boolean {
  if (sub.door < 0) return true;
  const reg = region(sub, new Set(boxes), player);
  return reg.set.has(sub.door);
}

// Binary heap ordered by (f, h, insertion counter): fully deterministic.
class Heap {
  a: Array<[number, number, number, string, number]> = [];
  get size(): number { return this.a.length; }
  push(e: [number, number, number, string, number]): void {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (Heap.less(a[i], a[p])) { const t = a[i]; a[i] = a[p]; a[p] = t; i = p; } else break;
    }
  }
  pop(): [number, number, number, string, number] | undefined {
    const a = this.a;
    if (a.length === 0) return undefined;
    const top = a[0];
    const last = a.pop() as [number, number, number, string, number];
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1; const r = l + 1;
        let m = i;
        if (l < a.length && Heap.less(a[l], a[m])) m = l;
        if (r < a.length && Heap.less(a[r], a[m])) m = r;
        if (m === i) break;
        const t = a[i]; a[i] = a[m]; a[m] = t; i = m;
      }
    }
    return top;
  }
  static less(x: [number, number, number, string, number], y: [number, number, number, string, number]): boolean {
    if (x[0] !== y[0]) return x[0] < y[0];
    if (x[1] !== y[1]) return x[1] < y[1];
    return x[2] < y[2];
  }
}

// Push-based search over box configurations. mode 'astar' pops by
// f = g + h (first goal popped is push-optimal); mode 'greedy' pops
// by h (finds any solution fast). Returns null when the cap or the
// shared budget runs out without a solution, except that a goal state
// generated along the way is returned as a (possibly suboptimal)
// fallback instead of nothing.
function solveSub(sub: Sub, mode: 'astar' | 'greedy', cap: number): SubResult | null {
  if (sub.startPlayer < 0 || sub.startBoxes.length === 0) return null;
  if (sub.startBoxes.length !== sub.goals.length) return null;
  const startBoxes = sub.startBoxes.slice();
  const startSet = new Set(startBoxes);
  if (isDead(sub, startBoxes, startSet)) return null;
  const startPrep = region(sub, startSet, sub.startPlayer).rep;
  const startKey = stateKey(startBoxes, startPrep);
  const bestG = new Map<string, number>();
  const boxesOf = new Map<string, number[]>();
  const playerOf = new Map<string, number>();
  const parent = new Map<string, ParentInfo>();
  bestG.set(startKey, 0);
  boxesOf.set(startKey, startBoxes);
  playerOf.set(startKey, sub.startPlayer);
  const heap = new Heap();
  let counter = 0;
  const h0 = heuristic(sub, startBoxes);
  if (h0 >= INF) return null;
  heap.push([h0, h0, counter++, startKey, 0]);
  let bestGoal: { key: string; g: number } | null = null;
  const considerGoal = (key: string, g: number, boxes: number[], player: number): void => {
    if (!isGoalState(sub, boxes)) return;
    if (sub.requireExit && !exitReachable(sub, boxes, player)) return;
    if (bestGoal === null || g < bestGoal.g) bestGoal = { key, g };
  };
  considerGoal(startKey, 0, startBoxes, sub.startPlayer);
  let expansions = 0;

  const reconstruct = (goalKey: string): SubResult | null => {
    const steps: PushStep[] = [];
    let cur = goalKey;
    while (parent.has(cur)) {
      const info = parent.get(cur) as ParentInfo;
      steps.push(info.step);
      cur = info.parent;
    }
    steps.reverse();
    let moves = '';
    let curPlayer = sub.startPlayer;
    let curBoxes = startBoxes.slice();
    for (const step of steps) {
      const behind = sub.nb[step.boxFrom][step.dir ^ 1];
      const path = walkPath(sub, new Set(curBoxes), curPlayer, behind);
      if (path === null) return null;
      moves += path + CH[step.dir];
      curPlayer = step.boxFrom;
      curBoxes = curBoxes.map((b) => (b === step.boxFrom ? step.boxTo : b));
      curBoxes.sort((a, b) => a - b);
    }
    return { pushes: steps.length, moves, finalPlayer: curPlayer };
  };

  while (heap.size > 0) {
    const entry = heap.pop() as [number, number, number, string, number];
    const g = entry[4];
    const key = entry[3];
    if (bestG.get(key) !== g) continue;
    if (expansions >= cap || BUDGET.used >= BUDGET.limit) {
      return bestGoal === null ? null : reconstruct(bestGoal.key);
    }
    expansions++;
    BUDGET.used++;
    const boxes = boxesOf.get(key) as number[];
    const player = playerOf.get(key) as number;
    if (isGoalState(sub, boxes) && (!sub.requireExit || exitReachable(sub, boxes, player))) {
      return reconstruct(key);
    }
    const boxSet = new Set(boxes);
    const reg = region(sub, boxSet, player);
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      for (let dir = 0; dir < 4; dir++) {
        const dest = sub.nb[b][dir];
        if (dest < 0 || !sub.room[dest] || boxSet.has(dest)) continue;
        const behind = sub.nb[b][dir ^ 1];
        if (behind < 0 || !reg.set.has(behind)) continue;
        const nb = boxes.slice();
        nb[i] = dest;
        nb.sort((x, y) => x - y);
        const nSet = new Set(nb);
        if (isDead(sub, nb, nSet)) continue;
        const nPrep = region(sub, nSet, b).rep;
        const nKey = stateKey(nb, nPrep);
        const ng = g + 1;
        if (ng >= (bestG.get(nKey) ?? INF)) continue;
        bestG.set(nKey, ng);
        boxesOf.set(nKey, nb);
        playerOf.set(nKey, b);
        parent.set(nKey, { parent: key, step: { boxFrom: b, boxTo: dest, dir } });
        considerGoal(nKey, ng, nb, b);
        const hh = heuristic(sub, nb);
        if (hh >= INF) continue;
        const f = mode === 'astar' ? ng + hh : hh;
        heap.push([f, hh, counter++, nKey, ng]);
      }
    }
  }
  return bestGoal === null ? null : reconstruct(bestGoal.key);
}

// Try the room searches in order: optimal with exit, optimal without
// exit, greedy without exit, greedy with exit.
function solveRoom(sub: Sub): SubResult | null {
  const tries: Array<[boolean, 'astar' | 'greedy']> = [
    [true, 'astar'], [false, 'astar'], [false, 'greedy'], [true, 'greedy'],
  ];
  for (const [requireExit, mode] of tries) {
    const s = { ...sub, requireExit, heurCache: sub.heurCache, dist: sub.dist };
    const r = solveSub(s, mode, mode === 'astar' ? ROOM_ASTAR_CAP : FALLBACK_CAP);
    if (r !== null) return r;
  }
  return null;
}

// ------------------------------------------------------------------
// Level decomposition: split at a separator row (the hallway) or
// column. Each resulting component must hold equal boxes and goals
// and touch the separator through exactly one door cell.
// ------------------------------------------------------------------
interface Room {
  cells: Array<[number, number]>;
  boxes: number[];
  goals: number[];
  doorKey: number;
  entryKey: number;
}

function decompose(level: Level, sepRow: number): Room[] | null {
  const seen = new Set<number>();
  const rooms: Room[] = [];
  let totalBoxes = 0;
  for (let r = 0; r < level.h; r++) {
    for (let c = 0; c < level.w; c++) {
      const k = keyOf(r, c);
      if (r === sepRow || level.walls.has(k) || seen.has(k)) continue;
      const stack = [k];
      seen.add(k);
      const comp: number[] = [];
      while (stack.length > 0) {
        const cur = stack.pop() as number;
        comp.push(cur);
        const cr = Math.floor(cur / 4096);
        const cc = cur % 4096;
        for (let d = 0; d < 4; d++) {
          const nr = cr + DR[d]; const nc = cc + DC[d];
          if (nr < 0 || nr >= level.h || nc < 0 || nc >= level.w || nr === sepRow) continue;
          const nk = keyOf(nr, nc);
          if (!level.walls.has(nk) && !seen.has(nk)) { seen.add(nk); stack.push(nk); }
        }
      }
      const boxes = comp.filter((x) => level.boxes.includes(x));
      const goals = comp.filter((x) => level.goals.has(x));
      if (boxes.length === 0 && goals.length === 0) continue;
      if (boxes.length === 0 || boxes.length !== goals.length) return null;
      const doors = comp.filter((x) => {
        const cr = Math.floor(x / 4096);
        const cc = x % 4096;
        for (const dr of [-1, 1]) {
          if (cr + dr === sepRow && !level.walls.has(keyOf(sepRow, cc))) return true;
        }
        return false;
      });
      if (doors.length !== 1) return null;
      const doorKey = doors[0];
      const dcc = doorKey % 4096;
      const entryKey = keyOf(sepRow, dcc);
      if (level.walls.has(entryKey)) return null;
      totalBoxes += boxes.length;
      rooms.push({
        cells: comp.map((x) => [Math.floor(x / 4096), x % 4096] as [number, number]),
        boxes, goals, doorKey, entryKey,
      });
    }
  }
  if (rooms.length === 0 || totalBoxes !== level.boxes.length) return null;
  return rooms;
}

// Walk on the full grid from `from` to `to`, avoiding walls and the
// cells in `blocked`. When `hallwayRow` is given, stay on that row.
function gridWalk(level: Level, from: number, to: number, blocked: Set<number>, hallwayRow: number | null): string | null {
  if (from === to) return '';
  const prev = new Map<number, [number, number]>();
  const q = [from];
  prev.set(from, [-1, -1]);
  let head = 0;
  while (head < q.length) {
    const cur = q[head++];
    const cr = Math.floor(cur / 4096);
    const cc = cur % 4096;
    for (let d = 0; d < 4; d++) {
      const nr = cr + DR[d]; const nc = cc + DC[d];
      if (nr < 0 || nr >= level.h || nc < 0 || nc >= level.w) continue;
      if (hallwayRow !== null && nr !== hallwayRow) continue;
      const nk = keyOf(nr, nc);
      if (level.walls.has(nk) || blocked.has(nk) || prev.has(nk)) continue;
      prev.set(nk, [cur, d]);
      if (nk === to) {
        let s = '';
        let node = to;
        while (node !== from) {
          const [p, dir] = prev.get(node) as [number, number];
          s = CH[dir] + s;
          node = p;
        }
        return s;
      }
      q.push(nk);
    }
  }
  return null;
}

// Replay a move string on the level; true when it is legal and ends
// with every box on a goal.
function replay(level: Level, moves: string): boolean {
  const boxes = new Set(level.boxes);
  let pr = Math.floor(level.player / 4096);
  let pc = level.player % 4096;
  for (const ch of moves) {
    const d = CH.indexOf(ch.toLowerCase());
    if (d < 0) return false;
    const nr = pr + DR[d]; const nc = pc + DC[d];
    if (nr < 0 || nr >= level.h || nc < 0 || nc >= level.w) return false;
    const nk = keyOf(nr, nc);
    if (level.walls.has(nk)) return false;
    if (boxes.has(nk)) {
      const br = nr + DR[d]; const bc = nc + DC[d];
      if (br < 0 || br >= level.h || bc < 0 || bc >= level.w) return false;
      const bk = keyOf(br, bc);
      if (level.walls.has(bk) || boxes.has(bk)) return false;
      boxes.delete(nk);
      boxes.add(bk);
    }
    pr = nr; pc = nc;
  }
  for (const b of boxes) if (!level.goals.has(b)) return false;
  return true;
}

interface SolvedRoom {
  sub: Sub;
  result: SubResult;
  entryKey: number;
  exitPossible: boolean;
  boxes: number[];
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.player < 0 || level.boxes.length === 0) return '';
  if (level.boxes.length !== level.goals.size) return '';

  const playerRow = Math.floor(level.player / 4096);
  let rooms: Room[] | null = decompose(level, playerRow);
  if (rooms === null) {
    for (let r = 0; r < level.h && rooms === null; r++) {
      if (r !== playerRow) rooms = decompose(level, r);
    }
  }
  if (rooms !== null) {
    const solved: SolvedRoom[] = [];
    for (const room of rooms) {
      const cells = room.cells.slice();
      cells.push([playerRow, room.entryKey % 4096]);
      const roomKeys = new Set(room.cells.map(([r, c]) => keyOf(r, c)));
      const sub = buildSub(cells, roomKeys, room.goals, room.boxes, room.entryKey, room.doorKey, room.entryKey, true);
      const result = solveRoom(sub);
      if (result === null) return '';
      const exitPossible = exitReachable(sub, resultFinalBoxes(sub, result), result.finalPlayer);
      solved.push({ sub, result, entryKey: room.entryKey, exitPossible, boxes: room.boxes });
    }
    // Rooms the player can leave go first (hardest first); a room the
    // player cannot leave, if any, goes last.
    solved.sort((a, b) => {
      if (a.exitPossible !== b.exitPossible) return a.exitPossible ? -1 : 1;
      return b.result.pushes - a.result.pushes;
    });
    let moves = '';
    let curPos = level.player;
    const allBoxes = new Set(level.boxes);
    for (let i = 0; i < solved.length; i++) {
      const s = solved[i];
      const last = i === solved.length - 1;
      let walk = gridWalk(level, curPos, s.entryKey, allBoxes, playerRow);
      if (walk === null) walk = gridWalk(level, curPos, s.entryKey, allBoxes, null);
      if (walk === null) return '';
      moves += walk;
      moves += s.result.moves;
      if (!last) {
        if (!s.exitPossible) return '';
        const finalBoxes = resultFinalBoxes(s.sub, s.result);
        const exit = walkPath(s.sub, new Set(finalBoxes), s.result.finalPlayer, s.sub.door);
        if (exit === null) return '';
        const [dr, dc] = s.sub.rc[s.sub.door];
        const [er, ec] = s.sub.rc[s.sub.entry];
        let dir = -1;
        for (let d = 0; d < 4; d++) {
          if (dr + DR[d] === er && dc + DC[d] === ec) { dir = d; break; }
        }
        if (dir < 0) return '';
        moves += exit + CH[dir];
        curPos = s.entryKey;
        // Boxes of the solved room moved from their starts to goals.
        for (const b of s.boxes) allBoxes.delete(b);
        for (const b of finalBoxes) allBoxes.add(keyOf(s.sub.rc[b][0], s.sub.rc[b][1]));
      }
    }
    if (!replay(level, moves)) return '';
    return moves;
  }

  // Fallback: whole-level search (used only if the level does not
  // decompose into rooms as expected).
  const allCells: Array<[number, number]> = [];
  for (let r = 0; r < level.h; r++) {
    for (let c = 0; c < level.w; c++) {
      if (!level.walls.has(keyOf(r, c))) allCells.push([r, c]);
    }
  }
  const allKeys = new Set(allCells.map(([r, c]) => keyOf(r, c)));
  const sub = buildSub(allCells, allKeys, [...level.goals], level.boxes, level.player, -1, -1, false);
  const result = solveSub(sub, 'astar', ROOM_ASTAR_CAP) ?? solveSub(sub, 'greedy', FALLBACK_CAP);
  if (result === null) return '';
  if (!replay(level, result.moves)) return '';
  return result.moves;
}

// Boxes of a subproblem after its solution: replay its pushes.
function resultFinalBoxes(sub: Sub, result: SubResult): number[] {
  // The solution ends with every box on a goal; the final box set is
  // exactly the goal set (as indices). Reconstructing it this way
  // avoids storing the push chain.
  return sub.goals.slice();
}

for (const id of readdirSync(INPUT).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
