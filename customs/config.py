batch_size = 1024
learning_rate = 1e-6
momentum = 0.9
numEpochs = 5
test_batch_size = 1536
numClasses = 200 # 10 for cifar-10, 200 for tinyimagenet
numImages = 2
numDenoisingSteps = 25

max_length = 25

rl_lr = 3e-4 
rl_batch = 512
rl_epochs = 5 
rl_steps = 2048
rl_timesteps = 2000


# CIFAR10 ResNet18/20/50/56

# pretrain_bs = 128
# pretrain_epochs = 200
# pretrain_lr = 1e-1
# pretrain_momentum = 0.9
# pretrain_weight_decay = 5e-4

# CIFAR10 VIT

pretrain_bs = 100
pretrain_epochs = 200
pretrain_lr = 1e-4
pretrain_momentum = 0.9
pretrain_weight_decay = 5e-4


classList = ['airplane', 'automobile', 'bird (animal)', 'cat (animal)', 'deer (animal)', 'dog (animal)', 'frog', 'horse (animal)', 'ship', 'truck']


# TINYIMAGENET EFFICIENTNET