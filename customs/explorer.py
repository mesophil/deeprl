import gymnasium as gym
import numpy as np
from gymnasium import spaces
import glob

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
        self.action_space = spaces.Dict({"first" : spaces.Discrete(10),
                                         "second" : spaces.Discrete(2),
                                         "third" : spaces.Discrete(2),
                                         "fourth" : spaces.Discrete(2)})

        self.classes = classes

        for c in self.classes:
            makeImage(c, c)

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

        for key in action:
            if key == 'first':
                phrase = " ".join([phrase, self.classes[action[key]]])
            elif key == 'second':
                phrase = " ".join([phrase, sizes[action[key]]])
            else:
                phrase = " ".join([phrase, vocab[action[key]]])
            
        dire = vocab[action[key]]

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

    env = trainEnv()
    check_env(env)

def testEnv():
    env = trainEnv()

    vec_env = make_vec_env(trainEnv, n_envs=1, env_kwargs=dict(maxLength=10))

    model = A2C("MlpPolicy", env, verbose=1).learn(50)

    obs = vec_env.reset()
    n_steps = 20
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, done, info = vec_env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)
        vec_env.render()
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break


if __name__ == "__main__":
    logging.basicConfig(filename='my.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.DEBUG)

    main()