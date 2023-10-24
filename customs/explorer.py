import gymnasium as gym
import numpy as np
from gymnasium import spaces
import glob

import logging, os
from processor import doImage, getInitialAcc

from stable_baselines3.common.env_checker import check_env


class trainEnv(gym.Env):

    def __init__(self, maxLength = 10) -> None:
        super(trainEnv).__init__()

        self.maxLength = maxLength
        self.currLength = 0
        self.action_space = spaces.Dict({"first" : spaces.Discrete(4),
                                         "second" : spaces.Discrete(4),
                                         "third" : spaces.Discrete(4)})

        '''
        H&E Stained Image with cancer
        {'normal', 'adeno', ...}
        {'lung', 'colon', 'breast' ...}
        {'lung', 'colon', 'breast' ...}
        ( or just have every word in every space )
        highly accurate highly detailed
        
        '''

        self.currentAcc = self.initialAcc = getInitialAcc()

        self.observation_space = spaces.Box(low = 0, high = 1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.currLength = 0
        self.currentAcc = self.initialAcc

        # clear the images folder
        currentDir = os.path.dirname(os.path.realpath(__file__))

        imgPath = os.path.join(currentDir, "../images")

        # for f in os.listdir(imgPath):
        #     if not f.endswith('.png'): continue
        #     os.remove(os.path.join(imgPath, f))

        pathsList = glob.glob(imgPath + '/**/*', recursive=True)

        for p in pathsList:
            if os.path.isfile(p) and p.endswith('.png'):
                os.remove(p)

        return np.array([self.currentAcc]).astype(np.float32), {} #empty dict is for info
    
    def step(self, action):
        reward = 0
        terminated, truncated = False, False
        info = {}

        phrase = "H&E stain pathology digital image"
        vocab = ['lung', 'colon', 'benign', 'adipose']

        for key in reversed(action):
            phrase = " ".join([phrase, vocab[action[key]]])
            
        dire = vocab[action[key]]

        phrase = " ".join([phrase, "highly detailed, highly accurate, colour correct"])

        # send
        # phrase
        # to
        # stable diffusion
        newAcc = doImage(phrase, dire)

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

    env = trainEnv()
    check_env(env)


if __name__ == "__main__":
    logging.basicConfig(filename='my.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.DEBUG)

    main()