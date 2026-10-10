import numpy as np
import pandas as pd

class WarehouseEnv:
    def __init__(self, bin_file, picklist_file, grid_size=10):
        self.bin_df = pd.read_csv(bin_file)
        self.pick_df = pd.read_csv(picklist_file)
        self.grid_size = grid_size
        
        self.bin_lookup = {
            row["bin_id"]: (row["x_coord"], row["y_coord"])
            for _, row in self.bin_df.iterrows()
        }

        self.all_bins = list(self.pick_df["bin_id"].unique())
        self.num_bins = len(self.all_bins)

        # ✅ Manhattan max distance
        self.max_distance = 2 * grid_size

    def reset(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.remaining_bins = self.all_bins.copy()
        self.total_distance = 0.0
        self.steps_taken = 0

        # ✅ Track visited
        self.visited = set()
        self.visited.add((self.current_x, self.current_y))

        return self._get_state()

    def _get_state(self):
        return np.array([
            self.current_x / self.grid_size,
            self.current_y / self.grid_size,
            self.steps_taken / self.num_bins,
            len(self.remaining_bins) / self.num_bins
        ], dtype=np.float32)

    def get_state_action_features(self):
        features = []

        norm_curr_x = self.current_x / self.grid_size
        norm_curr_y = self.current_y / self.grid_size
        progress = self.steps_taken / self.num_bins

        for bin_id in self.remaining_bins:
            x, y = self.bin_lookup[bin_id]

            # ✅ Manhattan distance
            dist = abs(self.current_x - x) + abs(self.current_y - y)
            norm_dist = dist / self.max_distance

            norm_target_x = x / self.grid_size
            norm_target_y = y / self.grid_size

            is_last = 1.0 if len(self.remaining_bins) == 1 else 0.0

            features.append(np.array([
                norm_curr_x,
                norm_curr_y,
                norm_target_x,
                norm_target_y,
                norm_dist,
                progress,
                is_last
            ], dtype=np.float32))

        return np.array(features)

    def step(self, action_idx):
        target_bin = self.remaining_bins[action_idx]
        next_x, next_y = self.bin_lookup[target_bin]

        # ✅ Previous nearest distance
        if self.remaining_bins:
            prev_dist = min([
                abs(self.current_x - self.bin_lookup[b][0]) +
                abs(self.current_y - self.bin_lookup[b][1])
                for b in self.remaining_bins
            ])
        else:
            prev_dist = 0

        # ✅ Manhattan movement
        dist = abs(self.current_x - next_x) + abs(self.current_y - next_y)

        self.total_distance += dist
        self.current_x, self.current_y = next_x, next_y
        self.remaining_bins.pop(action_idx)
        self.steps_taken += 1

        reward = -dist  # base penalty

        # ✅ Visited penalty
        if (next_x, next_y) in self.visited:
            reward -= 5
        self.visited.add((next_x, next_y))

        # ✅ Directional reward
        if self.remaining_bins:
            new_dist = min([
                abs(self.current_x - self.bin_lookup[b][0]) +
                abs(self.current_y - self.bin_lookup[b][1])
                for b in self.remaining_bins
            ])
            reward += (prev_dist - new_dist) * 2

        done = len(self.remaining_bins) == 0

        # ✅ Strong completion reward
        if done:
            reward += 200 - self.total_distance

        return self._get_state(), reward, done, {
            "total_distance": self.total_distance
        }

    def get_greedy_baseline(self):
        x, y = 0.0, 0.0
        bins = self.all_bins.copy()
        total = 0.0

        while bins:
            dists = [
                abs(x - self.bin_lookup[b][0]) +
                abs(y - self.bin_lookup[b][1])
                for b in bins
            ]
            idx = np.argmin(dists)
            total += dists[idx]
            x, y = self.bin_lookup[bins.pop(idx)]

        return total