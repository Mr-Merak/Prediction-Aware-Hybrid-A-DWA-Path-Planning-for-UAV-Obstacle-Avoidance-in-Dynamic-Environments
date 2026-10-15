import numpy as np
import matplotlib
import os
if os.environ.get('DISPLAY') or os.name == 'nt':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.animation import FuncAnimation
import math
from scipy.interpolate import splev, splprep


class UAV:
    def __init__(self, pos, radius=1.0, max_speed=2.0, max_turn_rate=0.8, detection_radius=30.0):
        self.pos = np.array(pos, dtype=float)
        self.heading = 0.0
        self.radius = radius
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        self.detection_radius = detection_radius
        self.v = 0.0
        self.omega = 0.0
        self.stuck_timer = 0
        self.prev_pos = self.pos.copy()

    def update(self, v, omega, dt):
        self.v = max(0.0, v)
        self.omega = max(-self.max_turn_rate, min(self.max_turn_rate, omega))
        self.heading += self.omega * dt
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        self.pos[0] += self.v * math.cos(self.heading) * dt
        self.pos[1] += self.v * math.sin(self.heading) * dt
        dist = math.hypot(self.pos[0] - self.prev_pos[0], self.pos[1] - self.prev_pos[1])
        if dist < 0.01:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0
            self.prev_pos = self.pos.copy()

    def distance_to(self, point):
        return math.hypot(self.pos[0] - point[0], self.pos[1] - point[1])


class KalmanFilter:
    def __init__(self, dt, process_noise=0.5, measurement_noise=2.0):
        self.dt = dt
        self.x = np.zeros(4)
        self.P = np.eye(4) * 5.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.Q[0, 0] = process_noise * dt ** 4 / 4
        self.Q[1, 1] = process_noise * dt ** 4 / 4
        self.Q[2, 2] = process_noise * dt ** 2
        self.Q[3, 3] = process_noise * dt ** 2
        self.R = np.eye(2) * measurement_noise
        self.initialized = False

    def initialize(self, obs_x, obs_y, obs_vx=0, obs_vy=0):
        self.x = np.array([obs_x, obs_y, obs_vx, obs_vy])
        self.P = np.eye(4) * 5.0
        self.initialized = True

    def predict(self, dt=None):
        if dt is not None:
            self.dt = dt
            self.F = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            self.Q = np.eye(4) * 0.5
            self.Q[0, 0] = 0.5 * dt ** 4 / 4
            self.Q[1, 1] = 0.5 * dt ** 4 / 4
            self.Q[2, 2] = 0.5 * dt ** 2
            self.Q[3, 3] = 0.5 * dt ** 2
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if not self.initialized:
            self.initialize(z[0], z[1])
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict_n_steps(self, n):
        F_n = np.array([[1, 0, n * self.dt, 0],
                        [0, 1, 0, n * self.dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        x_pred = F_n @ self.x
        P_pred = F_n @ self.P @ F_n.T + self.Q * n
        return x_pred, P_pred

    def get_predicted_pos_cov(self, n_steps):
        _, P_pred = self.predict_n_steps(n_steps)
        return P_pred[0:2, 0:2]


class ObstacleTracker:
    def __init__(self, obs_id, dt):
        self.obs_id = obs_id
        self.kf = KalmanFilter(dt)
        self.last_seen_time = 0
        self.is_tracked = False

    def update_observation(self, obs, current_time):
        z = np.array([obs.x, obs.y])
        self.kf.update(z)
        self.last_seen_time = current_time
        self.is_tracked = True

    def predict(self, dt):
        self.kf.predict(dt)

    def get_predicted_state(self, n_steps=1):
        return self.kf.predict_n_steps(n_steps)

    def get_position_covariance(self, n_steps=1):
        return self.kf.get_predicted_pos_cov(n_steps)


class Obstacle:
    def __init__(self, x, y, radius, risk_coeff=1.0, is_dynamic=False, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.risk_coeff = risk_coeff
        self.is_dynamic = is_dynamic
        self.vx = vx
        self.vy = vy

    def update(self, dt, bounds):
        if self.is_dynamic:
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.x < bounds[0] + self.radius:
                self.x = bounds[0] + self.radius
                self.vx = abs(self.vx)
            if self.x > bounds[1] - self.radius:
                self.x = bounds[1] - self.radius
                self.vx = -abs(self.vx)
            if self.y < bounds[2] + self.radius:
                self.y = bounds[2] + self.radius
                self.vy = abs(self.vy)
            if self.y > bounds[3] - self.radius:
                self.y = bounds[3] - self.radius
                self.vy = -abs(self.vy)

    def get_pos(self):
        return (self.x, self.y)


class Environment:
    def __init__(self, grid_size=(100, 100), resolution=2.0, bounds=(0, 200, 0, 200)):
        self.grid_size = grid_size
        self.resolution = resolution
        self.bounds = bounds
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.risk_grid = np.zeros(grid_size)
        self.obstacle_trackers = []

    def add_static_obstacle(self, x, y, radius, risk_coeff=1.0):
        self.static_obstacles.append(Obstacle(x, y, radius, risk_coeff, is_dynamic=False))

    def add_dynamic_obstacle(self, x, y, radius, vx, vy, risk_coeff=1.0):
        obs = Obstacle(x, y, radius, risk_coeff, is_dynamic=True, vx=vx, vy=vy)
        self.dynamic_obstacles.append(obs)
        tracker = ObstacleTracker(len(self.obstacle_trackers), dt=0.2)
        tracker.kf.initialize(x, y, vx, vy)
        self.obstacle_trackers.append(tracker)

    def update_dynamic(self, dt):
        for obs in self.dynamic_obstacles:
            obs.update(dt, self.bounds)

    def update_kf_trackers(self, current_time):
        for obs, tracker in zip(self.dynamic_obstacles, self.obstacle_trackers):
            z = np.array([obs.x, obs.y])
            tracker.update_observation(obs, current_time)

    def predict_kf_trackers(self, dt):
        for tracker in self.obstacle_trackers:
            tracker.predict(dt)

    def get_detected_obstacles(self, uav):
        detected_static = []
        for obs in self.static_obstacles:
            dist = math.hypot(obs.x - uav.pos[0], obs.y - uav.pos[1])
            if dist <= uav.detection_radius + obs.radius:
                detected_static.append(obs)
        detected_dynamic = []
        for obs in self.dynamic_obstacles:
            dist = math.hypot(obs.x - uav.pos[0], obs.y - uav.pos[1])
            if dist <= uav.detection_radius + obs.radius:
                detected_dynamic.append(obs)
        return detected_static, detected_dynamic

    def get_dynamic_tracker(self, obs):
        idx = None
        for i, o in enumerate(self.dynamic_obstacles):
            if o is obs:
                idx = i
                break
        if idx is not None and idx < len(self.obstacle_trackers):
            return self.obstacle_trackers[idx]
        return None

    def compute_risk_grid(self, uav, current_time, time_window=3.0, n_steps=5):
        self.risk_grid.fill(0)
        detected_static, detected_dynamic = self.get_detected_obstacles(uav)

        for obs in detected_dynamic:
            tracker = self.get_dynamic_tracker(obs)

            for step in range(1, n_steps + 1):
                if tracker and tracker.kf.initialized:
                    pred_state, pred_cov_full = tracker.get_predicted_state(step)
                    pred_x = pred_state[0]
                    pred_y = pred_state[1]
                    sigma = pred_cov_full[0:2, 0:2].copy()
                    sigma[0, 0] += obs.radius ** 2
                    sigma[1, 1] += obs.radius ** 2
                else:
                    pred_x = obs.x + obs.vx * step * 0.2
                    pred_y = obs.y + obs.vy * step * 0.2
                    sigma = np.eye(2) * (1.0 + step * 0.5)
                    sigma[0, 0] += obs.radius ** 2
                    sigma[1, 1] += obs.radius ** 2

                gx = int(pred_x / self.resolution)
                gy = int(pred_y / self.resolution)
                spread = max(2, int(math.sqrt(sigma[0, 0] + sigma[1, 1]) / self.resolution) + 1)

                for dx in range(-spread, spread + 1):
                    for dy in range(-spread, spread + 1):
                        ngx = gx + dx
                        ngy = gy + dy
                        if 0 <= ngx < self.grid_size[0] and 0 <= ngy < self.grid_size[1]:
                            px = ngx * self.resolution
                            py = ngy * self.resolution
                            diff = np.array([px - pred_x, py - pred_y])
                            try:
                                sigma_inv = np.linalg.inv(sigma)
                                mahalanobis = diff @ sigma_inv @ diff
                                risk = obs.risk_coeff * math.exp(-mahalanobis / 2.0)
                            except np.linalg.LinAlgError:
                                dist = math.hypot(px - pred_x, py - pred_y)
                                sigma_val = math.sqrt(sigma[0, 0] + sigma[1, 1])
                                risk = obs.risk_coeff * math.exp(-dist ** 2 / (2 * sigma_val ** 2 + 1e-6))
                            self.risk_grid[ngx, ngy] = max(self.risk_grid[ngx, ngy], risk)

        if np.max(self.risk_grid) > 0:
            max_risk = np.max(self.risk_grid)
            self.risk_grid = np.clip(self.risk_grid / max_risk, 0, 1)

    def is_collision(self, x, y, robot_radius, ignore_risk=False, risk_threshold=0.6):
        for obs in self.static_obstacles:
            if math.hypot(x - obs.x, y - obs.y) < robot_radius + obs.radius + 0.5:
                return True
        if not ignore_risk:
            gx = int(x / self.resolution)
            gy = int(y / self.resolution)
            if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
                if self.risk_grid[gx, gy] > risk_threshold:
                    return True
        return False

    def get_risk_at(self, x, y):
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
            return self.risk_grid[gx, gy]
        return 0.0


class AdaptiveAStar:
    def __init__(self, env, uav, goal, risk_threshold=0.6, k=10.0):
        self.env = env
        self.uav = uav
        self.goal = goal
        self.risk_threshold = risk_threshold
        self.k = k
        self.alpha_min, self.alpha_max = 0.5, 2.0
        self.beta_min, self.beta_max = 0.0, 2.0
        self.R0 = risk_threshold
        self.step_size = 4.0

    def heuristic(self, node, goal):
        return math.hypot(node[0] - goal[0], node[1] - goal[1])

    def adaptive_heuristic(self, node, goal, risk_val):
        basic_h = self.heuristic(node, goal)
        risk_h = risk_val * basic_h
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) / (1 + math.exp(self.k * (risk_val - self.R0)))
        beta = self.beta_max - (self.beta_max - self.beta_min) / (1 + math.exp(self.k * (risk_val - self.R0)))
        return alpha * basic_h + beta * risk_h

    def get_local_risk(self, node):
        x, y = node
        gx = int(x / self.env.resolution)
        gy = int(y / self.env.resolution)
        rad = 2
        total, cnt = 0, 0
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.env.grid_size[0] and 0 <= ny < self.env.grid_size[1]:
                    total += self.env.risk_grid[nx, ny]
                    cnt += 1
        return total / cnt if cnt > 0 else 0

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx = x + dx * self.step_size
            ny = y + dy * self.step_size
            if nx < self.env.bounds[0] or nx > self.env.bounds[1]:
                continue
            if ny < self.env.bounds[2] or ny > self.env.bounds[3]:
                continue
            if not self.env.is_collision(nx, ny, self.uav.radius, ignore_risk=False, risk_threshold=self.risk_threshold):
                neighbors.append((round(nx, 1), round(ny, 1)))
        return neighbors

    def run(self, max_iterations=5000):
        start = (round(self.uav.pos[0], 1), round(self.uav.pos[1], 1))
        goal = (round(self.goal[0], 1), round(self.goal[1], 1))

        open_set = {start}
        closed_set = set()
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        came_from = {}

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            current = min(open_set, key=lambda n: f_score[n])

            if math.hypot(current[0] - self.goal[0], current[1] - self.goal[1]) < self.step_size:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                step_cost = math.hypot(current[0] - neighbor[0], current[1] - neighbor[1])
                tentative_g = g_score[current] + step_cost
                if neighbor not in open_set or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    risk_val = self.get_local_risk(neighbor)
                    h_val = self.adaptive_heuristic(neighbor, goal, risk_val)
                    f_score[neighbor] = tentative_g + h_val
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        direct_path = [(self.uav.pos[0], self.uav.pos[1])]
        dist_to_goal = math.hypot(self.goal[0] - self.uav.pos[0], self.goal[1] - self.uav.pos[1])
        steps = max(int(dist_to_goal / 5.0), 5)
        for i in range(1, steps + 1):
            t = i / steps
            px = self.uav.pos[0] + t * (self.goal[0] - self.uav.pos[0])
            py = self.uav.pos[1] + t * (self.goal[1] - self.uav.pos[1])
            direct_path.append((px, py))
        return direct_path


class PathSmoother:
    @staticmethod
    def is_line_safe(env, p1, p2, risk_threshold=0.6, robot_radius=1.0):
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1e-6:
            return True
        steps = max(int(dist / env.resolution), 5)
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if env.is_collision(x, y, robot_radius, ignore_risk=False, risk_threshold=risk_threshold):
                return False
        return True

    @staticmethod
    def remove_redundant(path, env, risk_threshold=0.6):
        if len(path) < 3:
            return path
        optimized = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if PathSmoother.is_line_safe(env, path[i], path[j], risk_threshold):
                    optimized.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                i += 1
                if i < len(path):
                    optimized.append(path[i])
        return optimized

    @staticmethod
    def cubic_spline_smooth(path, num_points=100):
        if len(path) < 3:
            return path
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        try:
            tck, u = splprep([x, y], s=0, k=3)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            return [(x_new[i], y_new[i]) for i in range(num_points)]
        except Exception:
            return path


class DynamicWindowApproach:
    def __init__(self, uav, env, goal):
        self.uav = uav
        self.env = env
        self.goal = goal
        self.dt = 0.2
        self.predict_time = 1.5

    def calculate_velocity(self, global_path, local_goal):
        detected_static, detected_dynamic = self.env.get_detected_obstacles(self.uav)
        obstacles = [(obs.x, obs.y, obs.radius) for obs in detected_static]
        obstacles += [(obs.x, obs.y, obs.radius) for obs in detected_dynamic]

        goal_dir = (local_goal[0] - self.uav.pos[0], local_goal[1] - self.uav.pos[1])
        goal_dist = math.hypot(goal_dir[0], goal_dir[1])

        if goal_dist < 3.0:
            return self._go_towards(local_goal, self.uav.max_speed * 0.3)

        v_cruise = self.uav.max_speed * 0.8
        v_candidates = np.linspace(0.5, v_cruise, 12)
        omega_candidates = np.linspace(-self.uav.max_turn_rate, self.uav.max_turn_rate, 25)

        best_score = -float('inf')
        best_vel = (v_cruise, 0.0)
        has_valid = False

        for v in v_candidates:
            for omega in omega_candidates:
                traj = self._simulate_with_collision_check(v, omega, obstacles)
                if traj is None:
                    continue
                score = self._evaluate(traj, local_goal, global_path, obstacles)
                if score > best_score:
                    best_score = score
                    best_vel = (v, omega)
                    has_valid = True

        if not has_valid:
            best_vel = self._go_towards(local_goal, v_cruise * 0.6)
            safe = True
            test_traj = self._simulate_with_collision_check(best_vel[0], best_vel[1], obstacles)
            if test_traj is None:
                for omega in np.linspace(-self.uav.max_turn_rate, self.uav.max_turn_rate, 20):
                    test_v = self._go_towards(local_goal, v_cruise * 0.3)
                    trial = (v_cruise * 0.3, test_v[1] + omega)
                    trial = (trial[0], max(-self.uav.max_turn_rate, min(self.uav.max_turn_rate, trial[1])))
                    test_traj = self._simulate_with_collision_check(trial[0], trial[1], obstacles)
                    if test_traj is not None:
                        best_vel = trial
                        break

        return best_vel

    def _go_towards(self, target, speed=None):
        if speed is None:
            speed = self.uav.max_speed * 0.8
        dx = target[0] - self.uav.pos[0]
        dy = target[1] - self.uav.pos[1]
        desired_heading = math.atan2(dy, dx)
        omega = desired_heading - self.uav.heading
        omega = max(-self.uav.max_turn_rate, min(self.uav.max_turn_rate, omega))
        return (speed, omega)

    def _simulate_with_collision_check(self, v, omega, obstacles):
        if not obstacles:
            return [(self.uav.pos[0], self.uav.pos[1]), (self.uav.pos[0] + v * 0.3, self.uav.pos[1] + v * 0.3)]
        traj = [(self.uav.pos[0], self.uav.pos[1])]
        x, y = float(self.uav.pos[0]), float(self.uav.pos[1])
        heading = float(self.uav.heading)
        steps = int(self.predict_time / self.dt)

        for _ in range(steps):
            x += v * math.cos(heading) * self.dt
            y += v * math.sin(heading) * self.dt
            heading += omega * self.dt
            traj.append((x, y))

            for (ox, oy, obs_r) in obstacles:
                if math.hypot(x - ox, y - oy) < self.uav.radius + obs_r + 0.3:
                    return None

        return traj

    def _evaluate(self, traj, local_goal, global_path, obstacles):
        traj_end = traj[-1]
        dist_goal = math.hypot(traj_end[0] - local_goal[0], traj_end[1] - local_goal[1])

        heading_to_goal = math.atan2(local_goal[1] - self.uav.pos[1], local_goal[0] - self.uav.pos[0])
        traj_heading = math.atan2(traj_end[1] - traj[0][1], traj_end[0] - traj[0][0])
        heading_diff = abs(heading_to_goal - traj_heading)
        heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
        heading_score = 1.0 - heading_diff / math.pi

        min_obs_dist = 0.0
        safe_margin = 3.0
        for p in traj:
            for (ox, oy, obs_r) in obstacles:
                d = math.hypot(p[0] - ox, p[1] - oy) - self.uav.radius - obs_r
                min_obs_dist = min(min_obs_dist, d)

        if min_obs_dist < safe_margin:
            obs_score = 1.0 - min_obs_dist / safe_margin
        else:
            obs_score = 0.0

        w_heading = 1.0
        w_obs = 2.0
        w_goal = 0.5 / (dist_goal + 1.0)

        return w_heading * heading_score - w_obs * obs_score + w_goal


def main():
    bounds = (0, 200, 0, 200)
    env = Environment(grid_size=(100, 100), resolution=2.0, bounds=bounds)

    env.add_static_obstacle(40, 60, 8.0)
    env.add_static_obstacle(60, 40, 8.0)
    env.add_static_obstacle(70, 70, 6.0)
    env.add_static_obstacle(30, 30, 6.0)
    env.add_static_obstacle(50, 90, 10.0)
    env.add_static_obstacle(80, 50, 7.0)
    env.add_static_obstacle(100, 80, 5.0)
    env.add_static_obstacle(120, 100, 6.0)
    env.add_static_obstacle(140, 120, 5.0)
    env.add_static_obstacle(160, 60, 7.0)
    env.add_static_obstacle(110, 150, 8.0)
    env.add_static_obstacle(150, 180, 6.0)

    env.add_dynamic_obstacle(20, 80, 4.0, 1.2, -0.8)
    env.add_dynamic_obstacle(90, 30, 4.0, -0.8, 1.0)
    env.add_dynamic_obstacle(50, 50, 3.0, 0.6, 0.8)
    env.add_dynamic_obstacle(150, 150, 5.0, -1.0, 0.5)
    env.add_dynamic_obstacle(80, 120, 3.5, 0.8, -1.2)
    env.add_dynamic_obstacle(120, 30, 4.0, -0.5, 0.9)
    env.add_dynamic_obstacle(180, 80, 4.5, -1.2, -0.6)
    env.add_dynamic_obstacle(60, 180, 3.5, 1.0, -0.7)
    env.add_dynamic_obstacle(100, 20, 4.0, 0.7, 1.1)
    env.add_dynamic_obstacle(140, 60, 3.0, -0.9, -0.8)

    start = (10, 10)
    goal = (180, 180)
    uav = UAV(pos=start, radius=1.0, max_speed=2.5, max_turn_rate=0.9, detection_radius=35.0)
    uav.heading = math.atan2(goal[1] - start[1], goal[0] - start[0])

    env.compute_risk_grid(uav, 0.0)
    astar = AdaptiveAStar(env, uav, goal, risk_threshold=0.6)
    path = astar.run()
    print(f"Initial A* path: {len(path)} waypoints")

    smoother = PathSmoother()
    path = smoother.remove_redundant(path, env, risk_threshold=0.7)
    smooth_path = smoother.cubic_spline_smooth(path, num_points=150)

    dwa = DynamicWindowApproach(uav, env, goal)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title("UAV Dynamic Path Planning with Kalman Filter")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')

    for obs in env.static_obstacles:
        ax.add_patch(Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5))

    detection_circle = Circle((uav.pos[0], uav.pos[1]), uav.detection_radius,
                              fill=False, color='cyan', linewidth=1, linestyle='--', alpha=0.6)
    ax.add_patch(detection_circle)

    ax.plot(goal[0], goal[1], 'y*', markersize=20, label='Goal')

    global_line, = ax.plot([p[0] for p in smooth_path], [p[1] for p in smooth_path], 'b-', linewidth=2,
                           label='Global Path')

    local_traj_line, = ax.plot([], [], 'g--', linewidth=1.5, label='DWA Local Trajectory')

    kf_legend, = ax.plot([], [], 'm.', markersize=3, label='KF Predicted Path')

    dyn_circles = []
    for obs in env.dynamic_obstacles:
        c = Circle((obs.x, obs.y), obs.radius, color='blue', alpha=0.5)
        ax.add_patch(c)
        dyn_circles.append(c)

    local_vertices = np.array([[1.2, 0], [-0.6, 0.7], [-0.6, -0.7]])
    drone_poly = Polygon(local_vertices, closed=True, facecolor='green', edgecolor='darkgreen',
                         linewidth=1.5, alpha=0.9, label='UAV')
    ax.add_patch(drone_poly)

    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper right', fontsize=8)

    dt = 0.2
    sim_time = 0.0
    last_replan_time = 0.0
    replan_interval = 5.0
    risk_threshold_trigger = 0.3

    dwa_active = False
    dwa_start_time = 0.0
    dwa_consecutive_high_risk = 0

    kf_pred_patches = []

    trajectory_x = [uav.pos[0]]
    trajectory_y = [uav.pos[1]]
    traj_line, = ax.plot([], [], 'c-', linewidth=0.5, alpha=0.5, label='UAV Trajectory')

    def get_local_risk_around_uav(radius=5.0):
        total_risk = 0.0
        count = 0
        for dx in np.linspace(-radius, radius, 7):
            for dy in np.linspace(-radius, radius, 7):
                x = uav.pos[0] + dx
                y = uav.pos[1] + dy
                total_risk += env.get_risk_at(x, y)
                count += 1
        return total_risk / count if count > 0 else 0.0

    def replan():
        nonlocal smooth_path, global_line, kf_pred_patches

        for patch in kf_pred_patches:
            patch.remove()
        kf_pred_patches.clear()

        print(f"[{sim_time:.1f}s] Replanning from ({uav.pos[0]:.1f}, {uav.pos[1]:.1f})")

        env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time)

        astar_new = AdaptiveAStar(env, uav, goal, risk_threshold=0.6)
        new_path = astar_new.run()

        new_path = smoother.remove_redundant(new_path, env, risk_threshold=0.7)
        smooth_path = smoother.cubic_spline_smooth(new_path, num_points=150)

        global_line.set_data([p[0] for p in smooth_path], [p[1] for p in smooth_path])

        for obs in env.dynamic_obstacles:
            tracker = env.get_dynamic_tracker(obs)

            if tracker and tracker.kf.initialized:
                pred_positions = []
                pred_covs = []
                for step in range(1, 8):
                    ps, pc = tracker.get_predicted_state(step)
                    pred_positions.append((ps[0], ps[1]))
                    pred_covs.append(pc[0:2, 0:2])

                if pred_positions:
                    pred_x = [p[0] for p in pred_positions]
                    pred_y = [p[1] for p in pred_positions]
                    kf_patch, = ax.plot(pred_x, pred_y, 'm:', linewidth=1.5, markersize=3)
                    kf_pred_patches.append(kf_patch)

                    last_pred = pred_positions[-1]
                    cov = pred_covs[-1]
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
                        angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / math.pi
                        width = 2 * math.sqrt(eigenvalues[0]) * 2
                        height = 2 * math.sqrt(eigenvalues[1]) * 2
                        ellipse = Ellipse(last_pred, width, height, angle=angle,
                                          fill=False, color='magenta', linewidth=0.8, alpha=0.4, linestyle=':')
                        ax.add_patch(ellipse)
                        kf_pred_patches.append(ellipse)

        print(f"[{sim_time:.1f}s] Replanning done")

    def find_lookahead_target(path, current_pos, min_lookahead_dist=5.0, max_idx_offset=20):
        if not path:
            return goal

        min_dist = float('inf')
        closest_idx = 0
        for i, p in enumerate(path):
            d = math.hypot(current_pos[0] - p[0], current_pos[1] - p[1])
            if d < min_dist:
                min_dist = d
                closest_idx = i

        target = None
        for step in range(1, max_idx_offset + 1):
            idx = min(closest_idx + step, len(path) - 1)
            d = math.hypot(current_pos[0] - path[idx][0], current_pos[1] - path[idx][1])
            if d >= min_lookahead_dist:
                target = path[idx]
                break

        if target is None:
            target = path[-1]

        return target

    def animate(frame):
        nonlocal sim_time, last_replan_time, dwa_active, dwa_start_time, smooth_path, global_line
        nonlocal dwa_consecutive_high_risk, trajectory_x, trajectory_y

        sim_time += dt

        env.update_dynamic(dt)
        for i, obs in enumerate(env.dynamic_obstacles):
            dyn_circles[i].center = (obs.x, obs.y)

        env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time)

        detection_circle.center = (uav.pos[0], uav.pos[1])

        if uav.distance_to(goal) < 5.0:
            print(f"UAV reached goal at t={sim_time:.1f}s!")
            ani.event_source.stop()
            return [global_line, local_traj_line, drone_poly, detection_circle, info_text, traj_line] + dyn_circles + kf_pred_patches

        current_pos = tuple(uav.pos)

        replan_trigger = False
        if sim_time - last_replan_time >= replan_interval:
            replan_trigger = True

        if dwa_active and (sim_time - dwa_start_time > 3.0):
            current_risk = get_local_risk_around_uav()
            if current_risk < 0.15:
                dwa_active = False
                replan_trigger = True
                print(f"[{sim_time:.1f}s] DWA escaped, risk={current_risk:.3f}")

        if replan_trigger:
            replan()
            last_replan_time = sim_time

        local_risk = get_local_risk_around_uav()

        if local_risk > risk_threshold_trigger:
            dwa_consecutive_high_risk += 1
        else:
            dwa_consecutive_high_risk = 0

        if dwa_consecutive_high_risk >= 2 and not dwa_active:
            print(f"[{sim_time:.1f}s] High risk ({local_risk:.3f}), activating DWA")
            dwa_active = True
            dwa_start_time = sim_time
            dwa_consecutive_high_risk = 0

        if dwa_active:
            local_goal = find_lookahead_target(smooth_path, current_pos, min_lookahead_dist=3.0, max_idx_offset=10)
            v, omega = dwa.calculate_velocity(smooth_path, local_goal)

            active_traj = dwa._simulate_with_collision_check(
                v, omega,
                [(o.x, o.y, o.radius) for o in env.get_detected_obstacles(uav)[0]] +
                [(o.x, o.y, o.radius) for o in env.get_detected_obstacles(uav)[1]]
            )
            if active_traj:
                local_traj_line.set_data([p[0] for p in active_traj], [p[1] for p in active_traj])
            else:
                local_traj_line.set_data([], [])
        else:
            local_goal = find_lookahead_target(smooth_path, current_pos, min_lookahead_dist=8.0, max_idx_offset=25)
            v, omega = dwa.calculate_velocity(smooth_path, local_goal)
            local_traj_line.set_data([], [])

        uav.update(v, omega, dt)

        trajectory_x.append(uav.pos[0])
        trajectory_y.append(uav.pos[1])
        traj_line.set_data(trajectory_x, trajectory_y)

        cos_h = math.cos(uav.heading)
        sin_h = math.sin(uav.heading)
        verts = [(uav.pos[0] + v[0] * cos_h - v[1] * sin_h, uav.pos[1] + v[0] * sin_h + v[1] * cos_h) for v in
                 local_vertices]
        drone_poly.set_xy(verts)

        mode = "DWA AVOIDING" if dwa_active else "FOLLOWING PATH"
        info_str = (f"Time: {sim_time:.1f}s\n"
                    f"Mode: {mode}\n"
                    f"Risk: {local_risk:.3f}\n"
                    f"Speed: {v:.2f} m/s\n"
                    f"Dist: {uav.distance_to(goal):.1f}m\n"
                    f"Detect: {uav.detection_radius}m")
        info_text.set_text(info_str)

        return [global_line, local_traj_line, drone_poly, detection_circle, info_text, traj_line] + dyn_circles + kf_pred_patches

    ani = FuncAnimation(fig, animate, frames=1000, interval=50, blit=False, repeat=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
