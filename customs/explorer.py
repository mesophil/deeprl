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

        # CIFAR 10
        self.classes = classes

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

        self.maxLength = max(maxLength, len(self.classes))
        self.action_space = spaces.MultiDiscrete([len(self.classes), len(self.classes)])

        for c in self.classes:
            makeImage(c, c)

        self.initialAcc, self.initialClassAccuracies = getInitialAcc()
        self.currentAcc, self.classAccuracies = self.initialAcc, self.initialClassAccuracies

        logging.info(f"Initial accuracies: {self.initialAcc}, Class: {self.initialClassAccuracies}")

        self.observation_space = spaces.Box(low = 0, high = 1, shape=(12,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.currLength = 0
        self.currentAcc, self.classAccuracies = self.initialAcc, self.initialClassAccuracies

        # clear the images folder
        currentDir = os.path.dirname(os.path.realpath(__file__))

        imgPath = os.path.join(currentDir, "../images")

        pathsList = glob.glob(imgPath + '/**/*', recursive=True)

        for p in pathsList:
            if os.path.isfile(p) and p.endswith('.png'):
                os.remove(p)

        for p in pathsList:
            if os.path.isdir(p):
                os.rmdir(p)

        # initial seed (free)
        for c in self.classes:
            makeImage(c, c)

        obs = [self.currentAcc, self.currLength]
        obs.extend(self.classAccuracies)

        return np.array(obs).astype(np.float32), {} #empty dict is for info
    
    def step(self, action):
        reward = 0
        terminated, truncated = False, False
        info = {}

        phrase = "high quality hyperrealistic" # "an image easily confused between " class " and " class
        sizes = ["multiple", "large", "small"]
        vocab = ["on grass", "on sky", "on tree"]
        # [in snow, in rain, corrupted, blurry, fog]

        phrase = " ".join([self.classes[action[0]],
                           self.classes[action[1]],
                           phrase,
                           sizes[random.randint(0, 2)]])
        
        dire = self.classes[action[0]]

        phrase = " ".join([phrase, "highly detailed, highly accurate"])

        logging.info(f"Phrase: {phrase}")

        # send
        # phrase
        # to
        # stable diffusion
        newAcc, newClassAcc = doImage(phrase, dire)

        # add a reward equal to the accuracy improvement in percent
        # options: entropy, accuracy
        reward = (newAcc - self.currentAcc) * 100
        self.currLength += 1

        # terminate when the max number of images is reached
        # add early stoppage when validation accuracy stagnates
        terminated = self.currLength >= self.maxLength
        truncated = self.currentAcc < self.initialAcc

        self.currentAcc = newAcc

        obs = [self.currentAcc, self.currLength]
        obs.extend(newClassAcc)

        logging.info(f'Length: {self.currLength}, Acc: {newAcc}, Class Acc: {newClassAcc}')
        
        return (
            np.array(obs).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

def main():
    logging.info('start')

    testEnv()

def testEnv():
    env = trainEnv()

    vec_env = make_vec_env(trainEnv, n_envs=1, env_kwargs=dict(maxLength=10))

    logging.info(f"Training Model")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(10)

    obs = vec_env.reset()
    n_steps = 20
    
    logging.info(f"Using model")

    totalReward = 0

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        logging.info(f"Step {step + 1}")
        obs, reward, terminated, truncated, info = vec_env.step(action)
        totalReward += float(reward[0])
        logging.info(f"Reward: {reward[0]}")
        vec_env.render()
        if terminated or truncated:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            logging.info("Done")
            break
    
    logging.info(f"Total Reward: {totalReward}")


if __name__ == "__main__":
    logging.basicConfig(filename='my2.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)

    main()