import gymnasium as gym


class Game:
    def __init__(self):
        self.env = gym.make("CarRacing-v2")
        self.env.action_space.seed(42)
        self.seed=42

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation, info = self.env.reset()
        return observation, info

    def render(self):
        self.env.render()


