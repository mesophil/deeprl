import gymnasium as gym
import numpy as np
from gymnasium import spaces

import logging
from processor import doImage

class trainEnv(gym.Env):

    def __init__(self, maxLength = 10, initialAccuracy = 0) -> None:
        super().__init__()

        self.maxLength = maxLength
        self.currLength = 0
        self.action_space = spaces.Dict({"starter": 'H&E stain pathology digital image',
                                         "one" : ['lung', 'colon'],
                                         "two": ['lung', 'colon'],
                                         "three": ['benign', 'adipose'],
                                         "end": ['highly detailed, highly accurate']})

        '''
        H&E Stained Image with cancer
        {'lung', 'colon', 'breast' ...}
        {'lung', 'colon', 'breast' ...}
        {'normal', 'adeno', ...}
        highly accurate highly detailed
        
        '''

        self.initialAcc = initialAccuracy
        self.currentAcc = initialAccuracy

        self.observation_space = spaces.Box(low = 0, high = 1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.currLength = 0
        self.currentAcc = self.initialAcc

        return np.array([self.currentAcc]).astype(np.float32), {} #empty dict is for info
    
    def step(self, action):
        reward = 0
        terminated, truncated = False, False
        info = {}
        
        phrase = " ".join(action)

        # send
        # phrase
        # to
        # stable diffusion
        newAcc = doImage(phrase)

        # add a reward equal to the accuracy improvement in percent
        # options: entropy, accuracy
        reward = (newAcc - self.currentAcc) * 100
        self.currLength += 1

        # terminate when the max number of images is reached
        # add early stoppage when validation accuracy stagnates
        terminated = self.currLength >= self.maxLength

        self.currentAcc = newAcc
        
        return (
            np.array([self.currentAcc]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

def main():
    logging.info('start')


if __name__ == "__main__":
    logging.basicConfig(filename='my.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.DEBUG)

    main()