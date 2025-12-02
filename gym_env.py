import numpy as np
import gymnasium as gym
import random
import viewer_part

MAP_RANGE, DEPORT_DISTANCE, DEPORT_MARGIN = 0.5, 0.25, 0.25
STD_DEV = 0.25
REWARD_MIN, REWARD_MAX = 5, 20
SEED = 34   # Map random seed
SCALE = 25  # Coordinate magnification

class World(object):
    def __init__(self, depots, tasks, rewards, start_depot_index):
        self.depots = depots                        # warehouse point
        self.tasks = tasks                          # Delivery mission point
        self.rewards = rewards
        self.start_depot_index = start_depot_index  # Starting warehouse point

        self.points = np.vstack([depots, tasks])    # all points

        self.min_depot_dist = np.zeros(len(self.points), dtype=float)  # The distance from each point to the nearest depot
        for i, point in enumerate(self.points):
            dists = [np.linalg.norm(depot - point) for depot in self.depots]
            self.min_depot_dist[i] = np.min(dists)

    def reset(self, rewards, start_depot_index):    # Rebuilding map rewards and starting points
        self.rewards = rewards
        self.start_depot_index = start_depot_index

    def act(self, agent, action):                   # Changes to data for each move
        task = action - len(self.depots)
        if task < 0:
            reward = 0
        else:
            reward = self.rewards[task]

        dist = np.linalg.norm(self.points[agent] - self.points[action])

        return reward, dist

class PointsEnv(gym.Env):
    def __init__(self, rewards=None, start_depot_index=None, tasks_number=100, max_distance=3.0):
        self.fresh = True
        self.done = False

        self.depots_number = 4
        self.tasks_number = tasks_number
        self.max_distance = max_distance

        self.world = self.build_world(rewards, start_depot_index)
        self.viewer = None

        self.agent = self.world.start_depot_index
        self.remaining_distance = self.max_distance
        self.visited = np.zeros(len(self.world.points), dtype=int)  # Visited points
        self.path = np.array([], dtype=int)                         # Visited points, used for drawing
        self.path = np.append(self.path, self.agent)

    def check_distance(self, points):   # Verify whether the distance between warehouses meets the requirements
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < DEPORT_DISTANCE:
                    return False
        return True

    def GenerateGraph(self, tasks_number=100, seed=34):   # Generate map warehouse points and mission points
        state = np.random.get_state()
        np.random.seed(seed)

        x_min, x_max = 0, MAP_RANGE * 2
        y_min, y_max = 0, MAP_RANGE * 2

        x_split = np.linspace(x_min + DEPORT_MARGIN, x_max - DEPORT_MARGIN, 3)
        y_split = np.linspace(y_min + DEPORT_MARGIN, y_max - DEPORT_MARGIN, 3)

        depots = []

        for i in range(2):  # Randomly generate four warehouse points
            for j in range(2):
                # each quadrant range
                x_lo, x_hi = x_split[i], x_split[i + 1]
                y_lo, y_hi = y_split[j], y_split[j + 1]
                # Randomly select a point within the quadrant
                candidate = np.random.uniform([x_lo, y_lo], [x_hi, y_hi])
                depots.append(candidate)

        depots = np.array(depots)

        while not self.check_distance(depots):
            depots = []
            for i in range(2):
                for j in range(2):
                    x_lo, x_hi = x_split[i], x_split[i + 1]
                    y_lo, y_hi = y_split[j], y_split[j + 1]
                    candidate = np.random.uniform([x_lo, y_lo], [x_hi, y_hi])
                    depots.append(candidate)
            depots = np.array(depots)

        all_tasks = []
        tasks_per_depot = [tasks_number // len(depots)] * len(depots)
        for i in range(tasks_number % len(depots)):
            tasks_per_depot[i] += 1

        for i, (x, y) in enumerate(depots):
            count = tasks_per_depot[i]
            tasks = []
            while len(tasks) < count:
                tx = np.random.normal(x, STD_DEV)
                ty = np.random.normal(y, STD_DEV)
                if (x_min <= tx <= x_max) and (y_min <= ty <= y_max):
                    tasks.append([tx, ty])
            all_tasks = all_tasks + tasks

        all_tasks = np.array(all_tasks)

        np.random.set_state(state)
        return depots, all_tasks

    def build_world(self, rewards=None, start_depot_index=None):
        depots, tasks = self.GenerateGraph(self.tasks_number)
        if rewards is None:
            rewards = self.set_rewards()
        if start_depot_index is None:
            start_depot_index = self.set_start_depot_index()
        return World(depots, tasks, rewards, start_depot_index)

    def set_rewards(self):            # Set mission point rewards
        rewards = np.random.randint(REWARD_MIN, REWARD_MAX + 1, size=self.tasks_number)
        return rewards

    def set_start_depot_index(self):  # Set the robot starting warehouse
        start_depot_index = np.random.randint(0, self.depots_number)
        return start_depot_index

    def observe(self):
        depots = self.world.depots
        tasks = self.world.tasks
        rewards = self.world.rewards

        depot_rewards = np.zeros(len(depots))                            # [2]
        all_rewards = np.concatenate([depot_rewards, rewards])           # [D+T]

        # # Normalized to [-1,1]
        # all_rewards = 2.0 * all_rewards.astype(np.float32) / REWARD_MAX - 1.0
        # depots_norm = (depots - 0.5) * 2.0
        # tasks_norm = (tasks - 0.5) * 2.0

        # Normalized to [0,1]
        all_rewards[4:] = (all_rewards[4:].astype(np.float32) - REWARD_MIN) / (REWARD_MAX - REWARD_MIN)
        #print(all_rewards)

        # === 新增：已访问任务点奖励置零 ===
        # visited_mask = np.concatenate([np.zeros(0, dtype=bool), self.visited.astype(bool)])
        # #print(visited_mask)
        # all_rewards[visited_mask] = 0.0

        depots_norm = depots.copy()  # 如果 depots 本身已在 [0,1] 范围，可直接使用
        tasks_norm = tasks.copy()

        all_points = np.concatenate([depots_norm, tasks_norm], axis=0)   # [D+T, 2]

        graph = np.hstack([all_points, all_rewards[:, None]])            # [D+T, 3]
        agent = self.agent
        remaining_distance = self.remaining_distance
        valid_actions_mask = self.valid_actions_mask()
        return (graph,                                                   # All points and rewards
                agent,                                                   # Current point index
                remaining_distance,                                      # Remaining movable distance
                valid_actions_mask)

    # Get the mask of legal actions for the output mask of Transformer
    def valid_actions_mask(self):
        agent_pos = self.world.points[self.agent]
        remining_distance = self.remaining_distance
        dists = np.linalg.norm(agent_pos - self.world.points, axis=1)
        dists = dists + self.world.min_depot_dist
        valid_mask = (self.visited == 0) & (dists <= remining_distance)
        return valid_mask

    # Get all legal actions for
    def list_valid_actions(self):
        remaining_points = 1 - self.visited
        agent_pos = self.world.points[self.agent]
        remining_distance = self.remaining_distance
        dists = np.linalg.norm(agent_pos - self.world.points, axis=1)
        dists = dists + self.world.min_depot_dist
        remaining_points[dists > remining_distance] = 0
        return np.where(remaining_points)[0]

    def reset(self, rewards=None, start_depot_index=None):
        self.done = False
        self.fresh = True

        if rewards is None:
            rewards = self.set_rewards()
        if start_depot_index is None:
            start_depot_index = self.set_start_depot_index()
        self.world.reset(rewards, start_depot_index)

        if self.viewer is not None:
            self.viewer.reset()

        self.agent = self.world.start_depot_index
        self.remaining_distance = self.max_distance
        self.visited = np.zeros(len(self.world.points), dtype=int)  # Visited points
        self.path = np.array([], dtype=int)                         # Visited points, for drawing
        self.path = np.append(self.path, self.agent)

        return self.observe()

    # One-step execution of action, select point
    def step(self, action):
        self.fresh = False
        reward, dist = self.world.act(self.agent, action)

        self.visited[action] = 1
        self.path = np.append(self.path, action)
        self.agent = action
        self.remaining_distance -= dist
        if reward == 0:
            self.done = True

        observation = self.observe()

        return observation, reward, self.done

    def render(self, keep_open: bool = True):
        if self.viewer is None:
            self.viewer = viewer_part.Viewer()

        if self.viewer.fresh:
            self.viewer.fresh=False
            self.viewer.set_depots(self.world.depots)
            self.viewer.set_tasks(self.world.tasks, self.world.rewards)
            self.viewer.move_robot(self.path)
        else:
            self.viewer.move_robot(self.path)

        # Wait for animation to end
        while self.viewer.running:
            self.viewer.update()  # Manually execute a frame (automatically trigger update)
        if keep_open and hasattr(self, "viewer") and self.viewer is not None:
            if hasattr(self.viewer, "hold"):
                self.viewer.hold()
            else:
                import time
                print("渲染完成，窗口保持开启；按 Ctrl+C 退出")
                while getattr(self.viewer, "running", True):
                    self.viewer.update()
                    time.sleep(0.01)

if __name__ == "__main__":
    env = PointsEnv(tasks_number=100,max_distance=3.0)
    #env.reset()
    reward_sum = 0
    remaining_distance = 3.0
    done = False
    env.render()

    print(env.observe()[1])
    while not done:
        valid_actions = env.list_valid_actions()
        action = np.random.choice(valid_actions)

        # If there are still mission points to explore, do not return to the terminal - exhaust the endurance range
        while len(valid_actions) > 1 and action <= 3:
            action = np.random.choice(valid_actions)

        observation, reward, done = env.step(action)
        reward_sum += reward
        agent_pos = observation[1]
        remaining_distance = observation[2]

        print(observation[0])
        print(agent_pos)
        print(remaining_distance)

    env.render()

    print(f'OVER, and total reward is {reward_sum} and remaining distance is {remaining_distance}')
    env.viewer.hold()