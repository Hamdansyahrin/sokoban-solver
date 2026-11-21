// Sokoban solver: push-optimal A* per room, rooms composed over the hallway.
//
// A level from this generator is a hallway with independent rooms off it: a box pushed out of
// its room can never come back, so a level's push optimum is the sum of its rooms' optima.
// Each room is solved with A* over pushes (state: the boxes' squares plus the player's
// reachable region, named by its lowest square), guided by a min-cost matching of boxes to
// goals over true push distances (admissible, so the first solution found is push-optimal),
// pruned by dead squares, frozen boxes and solid 2x2 blocks. Budgets are expansion counts,
// never the clock, so two runs write the same bytes. A level whose rooms do not separate
// falls back to the same search over the whole level. Every answer is replayed locally
// before it is written; a level that cannot be solved gets an empty (valid) answer.
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const ROOM_BUDGET = 600_000;
const LEVEL_BUDGET = 2_000_000;
const BIG = 1_000_000;
// Directions: offset is applied later per width; here [dx, dy, walk letter, push letter].
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Min pushes to bring one box from every square to `goal`, other boxes ignored (pull BFS). */
function goalDist(wall: Uint8Array, w: number, size: number, goal: number): Int32Array {
  const dist = new Int32Array(size).fill(-1);
  dist[goal] = 0;
  const q = [goal];
  for (let qi = 0; qi < q.length; qi++) {
    const x = q[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (dist[y] < 0 && !wall[y] && !wall[y - d]) {
        dist[y] = dist[x] + 1;
        q.push(y);
      }
    }
  }
  return dist;
}

/** Squares from which a box can reach some goal at all (the dead-square complement). */
function liveSquares(wall: Uint8Array, w: number, size: number, goals: number[]): Uint8Array {
  const live = new Uint8Array(size);
  const q: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    q.push(g);
  }
  for (let qi = 0; qi < q.length; qi++) {
    const x = q[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        q.push(y);
      }
    }
  }
  return live;
}

/** Min-cost perfect matching (Hungarian, O(n^3)); BIG-ish totals mean no finite matching. */
function hungarian(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) if (p[j] > 0) total += cost[p[j] - 1][j - 1];
  return total;
}

/** The shortest walk from `from` to `to` around walls and boxes, lower LURD letters. */
function walkTo(w: number, wall: Uint8Array, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const p = q[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const nq = p + dy * w + dx;
      if (wall[nq] || occ[nq] || prev[nq] !== -1) continue;
      prev[nq] = p;
      how[nq] = letter;
      q.push(nq);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

interface Solved {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal A* on one board: `wall` already carries any foreign boxes as walls.
 * Returns the optimal solution or null when the expansion budget runs out.
 */
function solvePush(w: number, wall: Uint8Array, goals: number[], boxes0: number[], player0: number, budget: number): Solved | null {
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dists = goals.map((g) => goalDist(wall, w, size, g));
  const dead = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    if (wall[i]) continue;
    let alive = false;
    for (const d of dists) if (d[i] >= 0) { alive = true; break; }
    if (!alive) dead[i] = 1;
  }
  if (boxes0.some((b) => dead[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b] === 1);
  if (solved(boxes0)) return { pushes: 0, moves: '' };

  const hCache = new Map<string, number>();
  const heur = (boxes: number[]): number => {
    const key = String.fromCharCode(...boxes);
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    const cost = boxes.map((b) => dists.map((d) => (d[b] < 0 ? BIG : d[b])));
    const value = hungarian(cost);
    hCache.set(key, value);
    return value;
  };
  if (heur(boxes0) >= BIG) return null;

  // Player reachability scratch: flood fill with stamps.
  const seen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const region = (occ: Uint8Array, from: number): number => {
    stamp++;
    let low = from;
    let n = 0;
    stack[n++] = from;
    seen[from] = stamp;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      const q1 = p - w, q2 = p + w, q3 = p - 1, q4 = p + 1;
      if (!wall[q1] && !occ[q1] && seen[q1] !== stamp) { seen[q1] = stamp; stack[n++] = q1; }
      if (!wall[q2] && !occ[q2] && seen[q2] !== stamp) { seen[q2] = stamp; stack[n++] = q2; }
      if (!wall[q3] && !occ[q3] && seen[q3] !== stamp) { seen[q3] = stamp; stack[n++] = q3; }
      if (!wall[q4] && !occ[q4] && seen[q4] !== stamp) { seen[q4] = stamp; stack[n++] = q4; }
    }
    return low;
  };

  const occ = new Uint8Array(size);
  const occOf = (boxes: number[]): void => {
    occ.fill(0);
    for (const b of boxes) occ[b] = 1;
  };

  /** A solid 2x2 of walls/boxes holding a box off its goal, around the pushed box at t. */
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1, c = a + w, d = c + 1;
      if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
      if ((occ[a] && !goalSet[a]) || (occ[b] && !goalSet[b]) || (occ[c] && !goalSet[c]) || (occ[d] && !goalSet[d])) return true;
    }
    return false;
  };

  /** Recursive freeze: a box that can move on neither axis (walls and frozen boxes block) is frozen. */
  const frozenDead = (boxes: number[]): boolean => {
    const frozen = new Set<number>();
    let progress = true;
    while (progress) {
      progress = false;
      for (const b of boxes) {
        if (frozen.has(b)) continue;
        const blocked = (c: number): boolean => wall[c] === 1 || (occ[c] === 1 && frozen.has(c));
        if ((blocked(b - w) || blocked(b + w)) && (blocked(b - 1) || blocked(b + 1))) {
          frozen.add(b);
          progress = true;
        }
      }
    }
    for (const b of boxes) if (frozen.has(b) && !goalSet[b]) return true;
    return false;
  };

  // A* state store.
  const sBoxes: number[][] = [boxes0.slice()];
  const sPlayer: number[] = [player0];
  const sParent: number[] = [-1];
  const sG: number[] = [0];
  const best = new Map<string, number>();
  occOf(boxes0);
  const low0 = region(occ, player0);
  best.set(String.fromCharCode(low0, ...boxes0), 0);
  // Heap on [f, -g, stateIndex].
  const heap: [number, number, number][] = [[heur(boxes0), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const hpop = (): [number, number, number] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, negG, s] = hpop();
    const g = -negG;
    if (g !== sG[s]) continue;
    const boxes = sBoxes[s];
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= budget) return null;
    expanded++;
    occOf(boxes);
    const here = region(occ, sPlayer[s]);
    const hereStamp = stamp;
    // Candidate pushes first (the region marks are read before any child region is computed).
    const cand: [number, number, number][] = [];
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== hereStamp || wall[t] || occ[t] || dead[t]) continue;
        cand.push([bi, d, t]);
      }
    }
    for (const [bi, d, t] of cand) {
      const b = boxes[bi];
      const next = boxes.slice();
      next[bi] = t;
      next.sort((a, b2) => a - b2);
      occOf(next);
      if (frozenBlock(t) || frozenDead(next)) continue;
      const low = region(occ, b);
      const key = String.fromCharCode(low, ...next);
      const g2 = g + 1;
      const seen2 = best.get(key);
      if (seen2 !== undefined && seen2 <= g2) continue;
      const h = heur(next);
      if (h >= BIG) continue;
      best.set(key, g2);
      const idx = sBoxes.length;
      sBoxes.push(next);
      sPlayer.push(b);
      sParent.push(s);
      sG.push(g2);
      hpush([g2 + h, -g2, idx]);
    }
  }
  if (found < 0) return null;
  // Reconstruct: walk to each push's stand square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxes0;
  for (const s of path) {
    const nextBoxes = sBoxes[s];
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    occOf(boxes);
    moves += walkTo(w, wall, occ, player, from - d);
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The level's rooms: cut the hallway row's dead squares; each part holding boxes is a room. */
function roomsOf(level: Level): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const live = liveSquares(wall, w, size, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Replay moves on the level: {solved, player} — the local self-check before writing. */
function replay(level: Level, moves: string): { solved: boolean; player: number } {
  const { w, wall } = level;
  const goalSet = new Set(level.goals);
  const box = new Set(level.boxes);
  let player = level.player;
  for (const ch of moves) {
    const dir = DIRS.find(([, , letter, push]) => letter === ch || push === ch);
    if (!dir) return { solved: false, player };
    const d = dir[1] * w + dir[0];
    const next = player + d;
    if (wall[next]) return { solved: false, player };
    if (box.has(next)) {
      const beyond = next + d;
      if (wall[beyond] || box.has(beyond)) return { solved: false, player };
      box.delete(next);
      box.add(beyond);
    }
    player = next;
  }
  for (const b of box) if (!goalSet.has(b)) return { solved: false, player };
  return { solved: true, player };
}

/** The player's square after `moves` from `from` (the moves are legal where they were made). */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch.toLowerCase() === 'u' ? -w : ch.toLowerCase() === 'd' ? w : ch.toLowerCase() === 'l' ? -1 : 1;
  return p;
}

function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let moves = '';
    let player = level.player;
    let ok = true;
    const state = rooms.map((r) => ({ goals: r.goals.slice().sort((a, b) => a - b), boxes: r.boxes.slice().sort((a, b) => a - b), done: false }));
    for (const room of state) {
      const wall = level.wall.slice();
      for (const other of state) if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      const sol = solvePush(level.w, wall, room.goals, room.boxes, player, ROOM_BUDGET);
      if (sol === null) {
        ok = false;
        break;
      }
      moves += sol.moves;
      player = endOf(level.w, player, sol.moves);
      room.done = true;
    }
    if (ok && replay(level, moves).solved) return moves;
  }
  const sol = solvePush(level.w, level.wall, level.goals, level.boxes, level.player, LEVEL_BUDGET);
  if (sol !== null && replay(level, sol.moves).solved) return sol.moves;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const dir = join(I, id);
  let text: string;
  try {
    if (!statSync(dir).isDirectory()) continue;
    text = readFileSync(join(dir, 'level.txt'), 'utf8');
  } catch {
    continue;
  }
  const moves = solveLevel(parseLevel(text));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
