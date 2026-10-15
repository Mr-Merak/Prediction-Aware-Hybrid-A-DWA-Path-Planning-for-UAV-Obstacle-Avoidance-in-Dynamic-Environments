"""
dstar_lite.py -- D* Lite incremental replanner (Koenig & Likhachev 2002).

Used as a competitive global-replanning baseline. Unlike A*, D* Lite reuses the
previous search when obstacle costs change, so it can afford to replan at a high
frequency (every simulation step). It does NOT predict obstacle motion: it only
sees the *current* positions of the dynamic obstacles, which is precisely the
property that distinguishes it from the proposed prediction-aware framework.

The implementation follows the original formulation:
    S. Koenig, M. Likhachev. "D* Lite." AAAI-02, pp. 476--483, 2002.

Grid: 100 x 100 cells, resolution 2 m, 8-connectivity, edge cost = Euclidean
distance in metres. A cell is blocked if its centre lies within
(obs.radius + robot.radius + margin) of any obstacle centre -- the same margin
(0.5 m) used by the paper's collision criterion (Section 5.2).
"""

import heapq
import math

INF = float("inf")


class DStarLite:
    def __init__(self, grid_size, resolution, bounds, static_obstacles,
                 dynamic_obstacles, start_world, goal_world,
                 robot_radius=1.0, margin=0.5):
        self.nx, self.ny = grid_size
        self.res = resolution
        self.bounds = bounds
        self.robot_radius = robot_radius
        self.margin = margin

        self.goal = self._w2g(goal_world)
        self.start = self._w2g(start_world)
        self.last_start = self.start
        self.km = 0.0

        self.static_blocked = self._blocked_cells(static_obstacles)
        self.dynamic_blocked = set()

        # g(s)  : cost-to-goal estimate (consistent when = rhs)
        # rhs(s): one-step lookahead g
        self.g = {}
        self.rhs = {}

        # priority queue keyed by (min(g, rhs) + h(start, s) + km, min(g, rhs))
        self.heap = []
        self._counter = 0
        self._entry = {}   # cell -> currently valid heap key (stale detection)

        self.rhs[self.goal] = 0.0
        self._insert(self.goal)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _w2g(self, p):
        return (int(p[0] / self.res), int(p[1] / self.res))

    def _g2w(self, c):
        return (c[0] * self.res + self.res / 2.0, c[1] * self.res + self.res / 2.0)

    def _blocked_cells(self, obstacles):
        """Cells whose centre is within (r + robot.radius + margin) of an obstacle."""
        blocked = set()
        for (x, y, r) in obstacles:
            r_eff = r + self.robot_radius + self.margin
            gx0 = int(math.floor((x - r_eff) / self.res))
            gx1 = int(math.floor((x + r_eff) / self.res))
            gy0 = int(math.floor((y - r_eff) / self.res))
            gy1 = int(math.floor((y + r_eff) / self.res))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    if 0 <= gx < self.nx and 0 <= gy < self.ny:
                        cx, cy = self._g2w((gx, gy))
                        if math.hypot(cx - x, cy - y) <= r_eff:
                            blocked.add((gx, gy))
        return blocked

    def is_blocked(self, c):
        return c in self.static_blocked or c in self.dynamic_blocked

    def _neighbors(self, c):
        gx, gy = c
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.nx and 0 <= ny < self.ny:
                    out.append((nx, ny))
        return out

    def _cost(self, u, v):
        if self.is_blocked(v):
            return INF
        return math.hypot((u[0] - v[0]) * self.res, (u[1] - v[1]) * self.res)

    def _h(self, a, b):
        return math.hypot((a[0] - b[0]) * self.res, (a[1] - b[1]) * self.res)

    # ------------------------------------------------------------------ #
    # Priority queue (lazy deletion)
    # ------------------------------------------------------------------ #
    def _key(self, s):
        g = self.g.get(s, INF)
        rhs = self.rhs.get(s, INF)
        m = min(g, rhs)
        return (m + self._h(self.start, s) + self.km, m)

    def _insert(self, s):
        k = self._key(s)
        self._entry[s] = k
        heapq.heappush(self.heap, (k, self._counter, s))
        self._counter += 1

    def _top(self):
        while self.heap:
            k, _, s = self.heap[0]
            if self._entry.get(s) == k:
                return k, s
            heapq.heappop(self.heap)
        return None, None

    def _pop(self):
        k, s = self._top()
        if s is None:
            return None, None
        heapq.heappop(self.heap)
        del self._entry[s]
        return k, s

    def _update_key(self, u):
        self._entry.pop(u, None)
        self._insert(u)

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    def update_vertex(self, u):
        if self.is_blocked(u):
            # A blocked cell cannot lie on any path; carry no finite value.
            self.g.pop(u, None)
            self.rhs.pop(u, None)
            self._entry.pop(u, None)
            return
        if u != self.goal:
            best = INF
            for v in self._neighbors(u):
                c = self._cost(u, v)
                gv = self.g.get(v, INF)
                if c < INF and gv < INF and c + gv < best:
                    best = c + gv
            if best < INF:
                self.rhs[u] = best
            else:
                self.rhs.pop(u, None)
        self._entry.pop(u, None)
        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._insert(u)

    def compute_shortest_path(self, max_iter=200000):
        it = 0
        while it < max_iter:
            it += 1
            k_old, u = self._top()
            if u is None:
                return
            start_key = self._key(self.start)
            g_start = self.g.get(self.start, INF)
            r_start = self.rhs.get(self.start, INF)
            if not (k_old < start_key or g_start != r_start):
                return
            k_new = self._key(u)
            if k_old < k_new:
                self._update_key(u)
                continue
            self._pop()
            gu = self.g.get(u, INF)
            ru = self.rhs.get(u, INF)
            if gu > ru:
                self.g[u] = ru
                for s in self._neighbors(u):
                    self.update_vertex(s)
            else:
                self.g.pop(u, None)
                self.update_vertex(u)
                for s in self._neighbors(u):
                    self.update_vertex(s)

    def set_start(self, start_world):
        new_start = self._w2g(start_world)
        if new_start != self.start:
            self.km += self._h(self.start, new_start)
            self.start = new_start

    def update_dynamic_obstacles(self, dynamic_obstacles):
        """Re-mark cells occupied by dynamic obstacles and repair affected vertices."""
        new_blocked = self._blocked_cells(dynamic_obstacles)
        newly_blocked = new_blocked - self.dynamic_blocked
        newly_free = self.dynamic_blocked - new_blocked
        self.dynamic_blocked = new_blocked

        affected = set()
        for c in newly_blocked:
            self.g.pop(c, None)
            self.rhs.pop(c, None)
            self._entry.pop(c, None)
            for n in self._neighbors(c):
                affected.add(n)
        for c in newly_free:
            affected.add(c)
            for n in self._neighbors(c):
                affected.add(n)
        for c in affected:
            self.update_vertex(c)

    def extract_path(self):
        """Greedy descent from start following argmin (cost + g), to the goal."""
        path = []
        cur = self.start
        path.append(cur)
        seen = set()
        while cur != self.goal and len(path) <= self.nx * self.ny:
            best = None
            best_val = INF
            for v in self._neighbors(cur):
                c = self._cost(cur, v)
                gv = self.g.get(v, INF)
                if c < INF and gv < INF and c + gv < best_val:
                    best_val = c + gv
                    best = v
            if best is None or best in seen:
                break
            seen.add(best)
            cur = best
            path.append(cur)
        return [self._g2w(c) for c in path]

    def plan(self, start_world):
        """One planning round: move start, update obstacles, recompute, return path."""
        self.set_start(start_world)
        self.compute_shortest_path()
        return self.extract_path()
