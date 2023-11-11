import gymnasium as gym
import numpy as np
from gymnasium import spaces
import glob
import random

import logging, os
from processor import doImage, getInitialAcc
from make_image import makeImage

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env


class trainEnv(gym.Env):

    def __init__(self, 
                 classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 
                 maxLength = 10) -> None:
        
        super(trainEnv).__init__()

        self.maxLength = maxLength
        self.currLength = 0
        # self.action_space = spaces.Dict({"first" : spaces.Discrete(10),
        #                                  "second" : spaces.Discrete(2),
        #                                  "third" : spaces.Discrete(2),
        #                                  "fourth" : spaces.Discrete(2)})

        self.action_space = spaces.Discrete(10)

        self.classes = classes

        for c in self.classes:
            makeImage(c, c)

        self.currentAcc = self.initialAcc = getInitialAcc()

        logging.info(f"Initial accuracy: {self.initialAcc}")

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

        for p in pathsList:
            if os.path.isdir(p):
                os.rmdir(p)

        for c in self.classes:
            makeImage(c, c)

        return np.array([self.currentAcc]).astype(np.float32), {} #empty dict is for info
    
    def step(self, action):
        reward = 0
        terminated, truncated = False, False
        info = {}

        phrase = "High quality realistic photograph"
        sizes = ["multiple", "large"]
        vocab = ["on grass", "on sky", "on tree"]

        # for i in range(len(action)):
        #     # logging.info('i: ' + str(i))
        #     if i == 0:
        #         phrase = " ".join([phrase, self.classes[action[i]]])
        #     elif i == 1:
        #         phrase = " ".join([phrase, sizes[action[i]]])
        #     else:
        #         phrase = " ".join([phrase, vocab[action[i]]])

        # dire = self.classes[action[0]]

        phrase = " ".join([phrase, 
                           self.classes[action], 
                           sizes[random.randint(0, 1)], 
                           vocab[random.randint(0, 2)]])

        dire = self.classes[action]

        phrase = " ".join([phrase, "highly detailed, highly accurate"])

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

    # for testing the environment
    # env = trainEnv()
    # check_env(env)

    testEnv()

def testEnv():
    env = trainEnv()

    vec_env = make_vec_env(trainEnv, n_envs=1, env_kwargs=dict(maxLength=10))

    logging.info(f"Training Model")
    model = PPO("MlpPolicy", env, verbose=1).learn(20)

    obs = vec_env.reset()
    n_steps = 20
    
    logging.info(f"Using model")

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        logging.info(f"Step {step + 1}")
        obs, reward, done, info = vec_env.step(action)
        logging.info(f"Reward: {reward[0]}")
        vec_env.render()
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            logging.info("Done")
            break


if __name__ == "__main__":
    logging.basicConfig(filename='my2.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)

    main()