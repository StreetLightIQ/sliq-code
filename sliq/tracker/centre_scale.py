"""Python port of the app tracker (app/.../tracking/CentreScaleTracker.kt).

Boxes are normalised (x1, y1, x2, y2) in [0, 1]. Time is in seconds.
"""
import math
from dataclasses import dataclass

MAX_MISSING_S = 1.5
CENTRE_GATE = 0.16
MAX_TRACKS = 48


@dataclass
class Track:
    id: int
    box: tuple
    score: float
    first_t: float
    last_t: float
    matches: int = 1
    vx: float = 0.0
    vy: float = 0.0
    misses: int = 0
    score_sum: float = 0.0
    max_area_box: tuple = None

    def __post_init__(self):
        self.score_sum = self.score
        self.max_area_box = self.box


def _cx(b):
    return (b[0] + b[2]) / 2


def _cy(b):
    return (b[1] + b[3]) / 2


def _w(b):
    return max(0.0, b[2] - b[0])


def _h(b):
    return max(0.0, b[3] - b[1])


def hungarian(cost):
    """Rectangular min-cost assignment (rows <= cols), same algorithm as the app."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [next(j for j in range(1, m + 1) if p[j] == row + 1) - 1 for row in range(n)]


class CentreScaleTracker:
    def __init__(self, max_missing_s=MAX_MISSING_S, centre_gate=CENTRE_GATE, max_tracks=MAX_TRACKS):
        self.max_missing_s, self.centre_gate, self.max_tracks = max_missing_s, centre_gate, max_tracks
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.prev_t = -math.inf

    def cost(self, t, box, now):
        dt = now - t.last_t
        dist = math.hypot(_cx(box) - _cx(t.box) - t.vx * dt, _cy(box) - _cy(t.box) - t.vy * dt)
        gate = self.centre_gate + min(0.08, dt * 0.04)
        scale = (abs(math.log(max(_w(box), 1e-5) / max(_w(t.box), 1e-5)))
                 + abs(math.log(max(_h(box), 1e-5) / max(_h(t.box), 1e-5))))
        if dist > gate or scale > 4:
            return 1e6
        return dist / gate * 0.75 + scale * 0.10

    def update(self, boxes, scores, now):
        """-> (active, matched, completed) lists of Track."""
        if now <= self.prev_t:
            raise ValueError('frames must have increasing timestamps')
        self.prev_t = now
        completed = [t for t in self.tracks.values() if now - t.last_t > self.max_missing_s]
        for t in completed:
            del self.tracks[t.id]
        existing = list(self.tracks.values())
        nd = len(boxes)
        costs = [[self.cost(t, boxes[c], now) if c < nd else 1.0 for c in range(nd + len(existing))] for t in existing]
        assignment = hungarian(costs)
        used = [False] * nd
        matched = []
        for row, t in enumerate(existing):
            col = assignment[row]
            if col < nd and costs[row][col] < 1.0:
                box, score = tuple(boxes[col]), float(scores[col])
                used[col] = True
                dt = max(1e-3, now - t.last_t)
                t.vx = min(1.0, max(-1.0, 0.6 * t.vx + 0.4 * (_cx(box) - _cx(t.box)) / dt))
                t.vy = min(1.0, max(-1.0, 0.6 * t.vy + 0.4 * (_cy(box) - _cy(t.box)) / dt))
                t.box, t.score, t.last_t = box, score, now
                t.matches += 1
                t.misses = 0
                t.score_sum += score
                if _w(box) * _h(box) > _w(t.max_area_box) * _h(t.max_area_box):
                    t.max_area_box = box
                matched.append(t)
            else:
                t.misses += 1
        for i in range(nd):
            if not used[i] and len(self.tracks) < self.max_tracks:
                t = Track(self.next_id, tuple(boxes[i]), float(scores[i]), now, now)
                self.next_id += 1
                self.tracks[t.id] = t
                matched.append(t)
        return list(self.tracks.values()), matched, completed

    def flush(self):
        out = list(self.tracks.values())
        self.tracks.clear()
        return out
