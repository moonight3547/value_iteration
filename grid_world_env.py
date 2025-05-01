import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode='human', size=3):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Discrete(size**4)
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.P = {}
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.transition()
    def transition(self):
        """
        Precompute the transition matrix `env.P` for all states and actions.
        """
        for state in range(self.observation_space.n):
            # Decode state into agent and target positions
            agent_flat = state // (self.size**2)
            target_flat = state % (self.size**2)
            
            agent_x, agent_y = divmod(agent_flat, self.size)
            target_x, target_y = divmod(target_flat, self.size)
            
            self.P[state] = {}
            
            for action in range(self.action_space.n):
                # Calculate new agent position after action
                direction = self._action_to_direction[action]
                new_agent_x = np.clip(agent_x + direction[0], 0, self.size - 1)
                new_agent_y = np.clip(agent_y + direction[1], 0, self.size - 1)
                
                # Check if the new position is the target
                new_agent_at_target = (new_agent_x == target_x) and (new_agent_y == target_y)
                reward = 1 if new_agent_at_target else 0
                terminated = new_agent_at_target
                
                # Compute the new state
                new_agent_flat = new_agent_x * self.size + new_agent_y
                new_state = new_agent_flat * (self.size ** 2) + target_flat
                
                # Compute Manhattan distance for info
                distance = abs(new_agent_x - target_x) + abs(new_agent_y - target_y)

                if distance == 0:
                    info = True
                else:
                    info = False
                
                # Store the transition (deterministic, so prob=1.0)
                self.P[state][action] = [(1.0, new_state, reward, info)]
    def get_obs(self):
        # Encode agent and target positions into a single integer
        agent_x, agent_y = self._agent_location
        target_x, target_y = self._target_location
        
        # Flatten positions to 1D indices
        agent_flat = agent_x * self.size + agent_y
        target_flat = target_x * self.size + target_y
        
        # Combine into single observation integer
        return agent_flat * (self.size ** 2) + target_flat
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self.get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self.get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

            
from gym.envs.registration import register
register(
    id='GridWorld-v0',
    entry_point='__main__:GridWorldEnv',
    max_episode_steps=300,
)
from value_iteration import *
if __name__ == '__main__':
    env = gym.make('GridWorld-v0', size=6)
    VI = ValueIterationAgent(env, gamma=1, iters=10000, eval_iters=100, eps=1e-2, seed=233333)
    VI.run()
    print("Optimal Policy: ", VI.policy)
    print("Optimal Value: ", VI.value)


    done = False
    while not done:
        action = VI.policy[VI.env.get_obs()]  
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


