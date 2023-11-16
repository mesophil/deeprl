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
                 classes = ['default'], 
                 maxLength = 10) -> None:
        
        super(trainEnv).__init__()

        # CIFAR 10
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # CIFAR 100
        # self.classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        #                 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        #                 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        #                 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        #                 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        #                 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        #                 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        #                 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        #                 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        #                 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        self.currLength = 0
        self.numGenerated = {i:0 for i in self.classes}

        self.maxLength = max(maxLength, len(self.classes))
        self.action_space = spaces.Discrete(len(self.classes))

        for c in self.classes:
            makeImage(c, c)

        self.currentAcc = self.initialAcc = getInitialAcc()

        logging.info(f"Initial accuracy: {self.initialAcc}")

        self.observation_space = spaces.Box(low = 0, high = 1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.currLength = 0
        self.currentAcc = self.initialAcc
        self.numGenerated = {i:0 for i in self.classes}

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
        self.numGenerated[self.classes[action]] += 1

        # terminate when the max number of images is reached
        # add early stoppage when validation accuracy stagnates
        terminated = self.currLength >= self.maxLength

        self.currentAcc = newAcc

        logging.info(f'Acc: {newAcc}')
        
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