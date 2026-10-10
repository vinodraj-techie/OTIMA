import numpy as np
import pandas as pd

class WarehouseEnv:
    def __init__(self, bin_file, picklist_file, grid_size=10):
        self.bin_df = pd.read_csv(bin_file)
        self.pick_df = pd.read_csv(picklist_file)
        self.grid_size = grid_size
        
        # Fast lookup dictionary
        self.bin_lookup = {
            row["bin_id"]: (row["x_coord"], row["y_coord"])
            for _, row in self.bin_df.iterrows()
        }
        
        # Store all bins that need to be visited
        self.all_bins = list(self.pick_df["bin_id"].unique())
        self.num_bins = len(self.all_bins)
        
        # Max possible distance (diagonal of grid)
        self.max_distance = np.sqrt(2 * (grid_size ** 2))
        
    def reset(self):
        """Reset environment to starting state"""
        self.current_x = 0.0  # dispatch point
        self.current_y = 0.0
        self.remaining_bins = self.all_bins.copy()
        self.total_distance = 0.0
        self.steps_taken = 0
        return self._get_state()
    
    def _get_state(self):
        """
        Returns a simple state representation:
        [normalized_x, normalized_y, progress, num_remaining_normalized]
        
        This is NOT used directly in the improved DQN approach,
        but kept for compatibility.
        """
        return np.array([
            self.current_x / self.grid_size,
            self.current_y / self.grid_size,
            self.steps_taken / self.num_bins,
            len(self.remaining_bins) / self.num_bins
        ], dtype=np.float32)
    
    def get_state_action_features(self):
        """
        Build feature vectors for each possible next action.
        Returns: numpy array of shape (num_remaining_bins, feature_dim)
        
        Features for each bin:
        - Current normalized position (x, y)
        - Target normalized position (x, y)  
        - Normalized distance to target
        - Normalized progress (how many bins visited)
        - Is this the last bin? (binary)
        """
        features = []
        
        norm_curr_x = self.current_x / self.grid_size
        norm_curr_y = self.current_y / self.grid_size
        progress = self.steps_taken / self.num_bins
        
        for bin_id in self.remaining_bins:
            x, y = self.bin_lookup[bin_id]
            
            # Calculate distance
            dist = np.sqrt((self.current_x - x)**2 + (self.current_y - y)**2)
            norm_dist = dist / self.max_distance
            
            # Normalize target coordinates
            norm_target_x = x / self.grid_size
            norm_target_y = y / self.grid_size
            
            # Is this the last bin?
            is_last = 1.0 if len(self.remaining_bins) == 1 else 0.0
            
            feature_vector = np.array([
                norm_curr_x,
                norm_curr_y,
                norm_target_x,
                norm_target_y,
                norm_dist,
                progress,
                is_last
            ], dtype=np.float32)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def step(self, action_idx):
        """
        Take a step by visiting the bin at action_idx in remaining_bins.
        
        Returns:
            state: new state (for compatibility, not used in improved approach)
            reward: immediate reward
            done: whether episode is complete
            info: additional information (distance traveled this step)
        """
        if action_idx < 0 or action_idx >= len(self.remaining_bins):
            raise ValueError(f"Invalid action {action_idx}. Valid range: 0-{len(self.remaining_bins)-1}")
        
        # Get target bin
        target_bin = self.remaining_bins[action_idx]
        next_x, next_y = self.bin_lookup[target_bin]
        
        # Calculate distance
        dist = np.sqrt((self.current_x - next_x)**2 + (self.current_y - next_y)**2)
        
        # Update state
        self.total_distance += dist
        self.current_x, self.current_y = next_x, next_y
        self.remaining_bins.pop(action_idx)
        self.steps_taken += 1
        
        # Calculate reward
        # Negative distance as penalty + bonus for completion
        reward = -dist
        
        done = len(self.remaining_bins) == 0
        
        if done:
            # Bonus for completing the route
            # Scale bonus based on efficiency
            completion_bonus = 50.0
            # Penalize total distance
            efficiency_bonus = max(0, 100 - self.total_distance)
            reward += completion_bonus + efficiency_bonus
        
        info = {
            'step_distance': dist,
            'total_distance': self.total_distance,
            'remaining': len(self.remaining_bins)
        }
        
        return self._get_state(), reward, done, info
    
    def get_greedy_baseline(self):
        """
        Calculate greedy nearest-neighbor baseline for comparison.
        Returns total distance.
        """
        x, y = 0.0, 0.0
        bins = self.all_bins.copy()
        total_dist = 0.0
        
        while bins:
            # Find nearest bin
            min_dist = float('inf')
            nearest_idx = 0
            
            for i, bin_id in enumerate(bins):
                bx, by = self.bin_lookup[bin_id]
                dist = np.sqrt((x - bx)**2 + (y - by)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            # Move to nearest bin
            target_bin = bins.pop(nearest_idx)
            x, y = self.bin_lookup[target_bin]
            total_dist += min_dist
        
        return total_dist
    
    def render(self, save_path=None):
        """Optional: visualize the current state"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot all bins
            all_x = [self.bin_lookup[b][0] for b in self.all_bins]
            all_y = [self.bin_lookup[b][1] for b in self.all_bins]
            ax.scatter(all_x, all_y, c='lightgray', s=100, label='Visited', zorder=1)
            
            # Plot remaining bins
            if self.remaining_bins:
                rem_x = [self.bin_lookup[b][0] for b in self.remaining_bins]
                rem_y = [self.bin_lookup[b][1] for b in self.remaining_bins]
                ax.scatter(rem_x, rem_y, c='red', s=100, label='Remaining', zorder=2)
            
            # Plot current position
            ax.scatter([self.current_x], [self.current_y], c='blue', s=200, 
                      marker='*', label='Current', zorder=3)
            
            # Plot dispatch point
            ax.scatter([0], [0], c='green', s=200, marker='s', 
                      label='Dispatch', zorder=3)
            
            ax.set_xlim(-0.5, self.grid_size + 0.5)
            ax.set_ylim(-0.5, self.grid_size + 0.5)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f'Warehouse Route\nDistance: {self.total_distance:.2f} | Remaining: {len(self.remaining_bins)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for rendering")