import gymnasium as gym
import numpy as np
from gymnasium import spaces
import glob
import random
import math

import logging, os
#from processor import doImage, getInitialAcc, test
from tin_processor import doImage, getInitialAcc
from make_image import makeImage

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from config import rl_lr, rl_batch, rl_epochs, rl_steps, rl_timesteps, max_length, numClasses

from tin_info import ids, id_to_english


class trainEnv(gym.Env):

    def __init__(self, 
                 numClasses, 
                 maxLength) -> None:
        
        super(trainEnv).__init__()

        self.numClasses = numClasses

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

        self.maxLength = max(maxLength, self.numClasses)

        # changed to have only half the class in each box for speed (just for testing)
        self.action_space = spaces.MultiDiscrete([self.numClasses, self.numClasses])


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

        self.init_classes()

        self.initialAcc, self.initialClassAccuracies = getInitialAcc()
        self.currentAcc, self.classAccuracies = self.initialAcc, self.initialClassAccuracies

        self.currEntropy = self.entropy(self.initialClassAccuracies)

        self.obsSize = 2 + self.numClasses

        logging.info(f"Initial accuracies: {self.initialAcc}, Entropy: {self.currEntropy}, Class: {self.initialClassAccuracies}")

        self.observation_space = spaces.Box(low = 0, high = 1, shape=(self.obsSize,), dtype=np.float32)

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
        self.init_classes()

        obs = [self.currentAcc, self.currLength]
        obs.extend(self.classAccuracies)

        self.currEntropy = self.entropy(self.initialClassAccuracies)

        return np.array(obs).astype(np.float32), {} #empty dict is for info
    
    def step(self, action):
        reward = 0
        terminated, truncated = False, False
        info = {}

        c = [id_to_english[ids[action[0]]], id_to_english[ids[action[1]]]]

        # c = [classList[action[0]], classList[action[1]]] for cifar-10

        # [in snow, in rain, corrupted, blurry, fog]
        domains = ['photo', 'drawing', 'painting', 'digital art image']

        phrase = " ".join(["A",
                           domains[np.random.randint(0, 3)],
                           "of a",
                           c[0],
                           "and",
                           c[1]
                           ])
        
        dire = str(action[0])

        # for ablation
        # phrase = c[0]
        # phrase = self.classes[np.random.randint(0, 9)]

        logging.info(f"Phrase: {phrase}")

        # send
        # phrase
        # to
        # stable diffusion
        newAcc, newClassAcc = doImage(phrase, dire)

        # add a reward equal to the accuracy improvement on the test/validation set
        # options: entropy, accuracy
        #reward = (newAcc - self.currentAcc)*100
        entpy = self.entropy(newClassAcc)
        reward = self.currEntropy - entpy
        self.currLength += 1

        self.currEntropy = entpy

        # terminate when the max number of images is reached
        # add early stoppage when validation accuracy stagnates
        terminated = self.currLength >= self.maxLength
        truncated = self.currentAcc < self.initialAcc - 0.005

        self.currentAcc = newAcc

        obs = [self.currentAcc, self.currLength]
        obs.extend(newClassAcc)

        logging.info(f'Length: {self.currLength}, Acc: {newAcc}, Ent: {entpy}, Class Acc: {newClassAcc}')
        
        return (
            np.array(obs).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )
    
    def init_classes(self):
        pass
        # for c in self.classes:
        #     if (c == 'bird' or c == 'cat' or c == 'deer' or c == 'dog' or c == 'frog' or c == 'horse'):
        #         makeImage(" ".join([c, '(animal)']), c)
        #     else:
        #         makeImage(c, c)
    
    def entropy(self, labels, base=None):
        """ Computes entropy of label distribution. """

        n_labels = len(labels)

        if n_labels <= 1:
            return 0

        value,counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.

        # Compute entropy
        base = math.e if base is None else base
        for i in probs:
            ent -= i * math.log(i, base)

        return ent


def main():
    logging.info('start')

    testEnv()

def testEnv():
    env = trainEnv(numClasses=numClasses, maxLength=max_length)

    vec_env = make_vec_env(trainEnv, n_envs=1, env_kwargs=dict(numClasses = numClasses, maxLength=max_length))

    checkpoint_callback = CheckpointCallback(save_freq=1000,
                                             save_path="./model_logs/",
                                             name_prefix="rl_model_tin_nolim",
                                             save_replay_buffer=True,
                                             save_vecnormalize=True,
                                             )

    logging.info(f"Training Model")
    model = PPO(policy="MlpPolicy", 
                env=env, 
                learning_rate=rl_lr, 
                batch_size=rl_batch, 
                n_epochs=rl_epochs, 
                n_steps=rl_steps, 
                ent_coef=0.01,
                vf_coef=0.5,
                verbose=1
                )
    
    model.learn(total_timesteps=rl_timesteps, callback=checkpoint_callback)

    obs = vec_env.reset()
    n_steps = 1000
    
    logging.info(f"Using model")

    totalReward = 0

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        logging.info(f"Step {step + 1}")
        obs, reward, terminated, _ = vec_env.step(action)
        totalReward += float(reward[0])
        logging.info(f"Reward: {reward[0]}")
        vec_env.render()
        if terminated:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            logging.info("Done")
            break
    
    logging.info(f"Total Reward: {totalReward}")


if __name__ == "__main__":
    logging.basicConfig(filename='my2.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)

    main()