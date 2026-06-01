import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, TYPE_CHECKING, Union
from extension.base import State, Action
from extension.goal import TerminalRegion
from extension.env import GridWorldEnv

if TYPE_CHECKING:
    from extension.policy import WVFMultiGoalAgent
    from extension.proposition import Proposition
    from extension.algebra import QFunction
    from extension.executor import EdgePolicy

class GridWorldRenderer:
    def __init__(self, env: GridWorldEnv, terminal_regions: List[TerminalRegion]):
        self.env = env
        self.terminal_regions = terminal_regions
        # Assign a color to each goal proposition ID
        self.colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        self.goal_colors = {
            goal.id: self.colors[i % len(self.colors)] 
            for i, goal in enumerate(terminal_regions)
        }

    def _setup_axes(self, ax):
        """Sets up the common grid properties for the axes."""
        # Set limits to cover the full range of cells [min, max + 1]
        ax.set_xlim(self.env.x_min, self.env.x_max + 1)
        ax.set_ylim(self.env.y_min, self.env.y_max + 1)
        ax.set_aspect('equal')
        # Ticks align with cell boundaries
        ax.set_xticks(range(self.env.x_min, self.env.x_max + 2))
        ax.set_yticks(range(self.env.y_min, self.env.y_max + 2))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _draw_base_grid(self, ax):
        """Draws the empty grid cells."""
        for x in range(self.env.x_min, self.env.x_max + 1):
            for y in range(self.env.y_min, self.env.y_max + 1):
                cell_rect = patches.Rectangle(
                    (x, y), 1, 1, 
                    linewidth=1, edgecolor='gray', facecolor='white', alpha=0.3
                )
                ax.add_patch(cell_rect)

    def _draw_goals(self, ax, outline_only: bool = False):
        """Draws the goal regions."""
        for x in range(self.env.x_min, self.env.x_max + 1):
            for y in range(self.env.y_min, self.env.y_max + 1):
                eval_state = State(x=x, y=y)
                for region in self.terminal_regions:
                    if region.contains(eval_state):
                        if outline_only:
                            # Draw a thick border so the heatmap underneath is visible
                            goal_rect = patches.Rectangle(
                                (x, y), 1, 1, 
                                linewidth=3, edgecolor=self.goal_colors[region.id], 
                                facecolor='none', alpha=1.0, zorder=10
                            )
                        else:
                            goal_rect = patches.Rectangle(
                                (x, y), 1, 1, 
                                facecolor=self.goal_colors[region.id], alpha=0.5
                            )
                        ax.add_patch(goal_rect)

    def _draw_policy_arrows(self, ax, policy: Union['QFunction', 'EdgePolicy']):
        """Draws arrows indicating the greedy action at each cell."""
        from extension.executor import EdgePolicy
        is_edge = isinstance(policy, EdgePolicy)
        
        action_map = {
            Action.UP: (0, 0.4),
            Action.DOWN: (0, -0.4),
            Action.LEFT: (-0.4, 0),
            Action.RIGHT: (0.4, 0)
        }
        
        for x in range(self.env.x_min, self.env.x_max + 1):
            for y in range(self.env.y_min, self.env.y_max + 1):
                state = State(x=x, y=y)
                
                if is_edge:
                    best_action = policy.get_action(state)
                else:
                    # Find best action
                    q_values = {a: policy(state, a) for a in Action}
                    # Only draw if there's meaningful learning (some values != 0)
                    if all(q == 0 for q in q_values.values()):
                        continue
                        
                    best_action = max(q_values, key=q_values.get)
                
                dx, dy = action_map[best_action]
                
                # Draw arrow centered in cell
                ax.arrow(
                    x + 0.5 - dx/2, y + 0.5 - dy/2, dx, dy,
                    head_width=0.2, head_length=0.2, fc='black', ec='black', 
                    alpha=0.6, zorder=15
                )

    def render(self, agent_state: State, history: List[State] = None, path_length: int = 5, save_path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        self._setup_axes(ax)
        self._draw_base_grid(ax)
        self._draw_goals(ax)

        # Draw Path (last N steps)
        if history and len(history) > 1:
            # Only take the last path_length steps
            path_coords = history[-(path_length+1):] + [agent_state]
            xs = [s.x + 0.5 for s in path_coords]
            ys = [s.y + 0.5 for s in path_coords]
            ax.plot(xs, ys, color='black', linewidth=1, linestyle='-', alpha=0.6, zorder=4)

        # Draw Agent Position centered in the cell
        agent_dot = plt.Circle((agent_state.x + 0.5, agent_state.y + 0.5), 0.3, color='black', zorder=5)
        ax.add_patch(agent_dot)
        
        ax.set_title(f"GridWorld ({agent_state.x}, {agent_state.y})")
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def render_policy(self, q_function: 'QFunction', title: str, save_path: str = None) -> None:
        """Visualizes the greedy policy as arrows on a clean grid."""
        fig, ax = plt.subplots(figsize=(8, 8))
        self._setup_axes(ax)
        self._draw_base_grid(ax)
        self._draw_goals(ax, outline_only=True)
        self._draw_policy_arrows(ax, q_function)
        
        ax.set_title(f"Policy: {title}")
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def render_value_function(self, q_function: 'QFunction', title: str, show_policy: bool = False, save_path: str = None) -> None:
        """Visualizes the learned World Value Function heatmap, optionally with policy arrows."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Compute Value Matrix V(s) = max_a Q(s, a)
        width = self.env.x_max - self.env.x_min + 1
        height = self.env.y_max - self.env.y_min + 1
        V = np.zeros((height, width))
        
        for x in range(self.env.x_min, self.env.x_max + 1):
            for y in range(self.env.y_min, self.env.y_max + 1):
                state = State(x=x, y=y)
                # Compute max Q value for this state using the generic QFunction
                q_values = [q_function(state, a) for a in Action]
                V[y - self.env.y_min, x - self.env.x_min] = max(q_values)
                
        # 2. Draw Heatmap
        im = ax.imshow(
            V, origin='lower', 
            extent=[self.env.x_min, self.env.x_max + 1, self.env.y_min, self.env.y_max + 1], 
            cmap='viridis', alpha=0.8
        )
        fig.colorbar(im, ax=ax, label='Value (Max Q)')
        
        # 3. Overlay context
        self._setup_axes(ax)
        self._draw_goals(ax, outline_only=True)
        
        if show_policy:
            self._draw_policy_arrows(ax, q_function)
            
        ax.set_title(f"Learned WVF: {title}")
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
