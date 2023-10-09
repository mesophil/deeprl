import gymnasium as gym
import numpy as np
from gymnasium import spaces

import logging

class trainEnv(gym.Env):

    def __init__(self, maxLength = 10) -> None:
        super().__init__()

        self.maxLength = maxLength
        
        self.action_space = spaces.Dict()

        self.observation_space = spaces.Box(low = 0, high = 1, shape=(1,), dtype=np.float32)


def main():
    logging.info('start')


if __name__ == "__main__":
    logging.basicConfig(filename='my.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.DEBUG)

    main()