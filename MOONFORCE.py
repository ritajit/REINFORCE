from collections import deque
import tensorflow as tf
import numpy as np
import gym
import os


####### Settings and Hyperparameters ######
ENV_NAME = 'LunarLander-v2'
RNG_SEED = 0

SAVE_INTERVAL = 100
MODEL_DIR = './models/'
SUMMARY_DIR = './tensorboard/REINFORCE/1'
MEANS_CSV_FILE = 'performance.csv'

RENDER = True

LEARNING_RATE = 0.005
GAMMA = 0.99

NUM_LEARN_EPISODES = 5000
NUM_PLAY_EPISODES = 300
###########################################

# the word 'mean' is used as a shorthand
# for referring to the the mean of the
# rewards of the last 100 episodes.
# the CartPole-v0 environment is said to be
# 'solved' when this value is above 196.

# seed numpy.random -- numpy.random.choice is
# used to sample actions from the action
# probability distribution
# seed tensorflow -- set graph-level random seed
np.random.seed(RNG_SEED)
tf.set_random_seed(RNG_SEED)

# class for an RL agent based on the
# REINFORCE algorithm
# (Monte-Carlo Policy Gradient)
class Agent():
    def __init__(self, env, LR = 0.01, GAMMA = 0.95):

        self.LR = LR
        self.GAMMA = GAMMA
        self.num_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        self.model = self.build_network()

        # saver for tensorflow sessions
        self.saver = tf.train.Saver()
        # create a new tensorflow session
        self.sess = tf.Session()
        # initialize tensorflow variables
        self.sess.run(tf.global_variables_initializer())
        # summary writer for tensorboard
        self.writer = tf.summary.FileWriter(SUMMARY_DIR)

        # store states, actions and rewards of one episode
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # agent's internal counter for episodes
        # to keep track of when to save the model
        self.episode = 0

    #store state, action and reward of one timestep
    def store_transition(self, S, A, R):

        self.episode_states.append(S)

        #create one-hot encoded array from action-index...
        action_ = np.zeros(self.num_actions)
        action_[A] = 1
        #...then store the array
        self.episode_actions.append(action_)

        self.episode_rewards.append(R)

    # create a normalized array containing the discounted returns from each
    # timestep in respective indices
    def normalized_discounted_returns(self):
        disc_returns = np.zeros_like(self.episode_rewards)
        timesteps = range(len(self.episode_rewards))
        reward_sum = 0
        # calculate and store the discounted returns
        # discounted return for timestep 't' is stored in index 't'
        for t in reversed(timesteps):
            reward_sum = self.episode_rewards[t] + self.GAMMA*reward_sum
            disc_returns[t] = reward_sum

        #normalize the array
        disc_returns -= np.mean(disc_returns)
        disc_returns /= np.std(disc_returns)
        return disc_returns

    # create the model
    def build_network(self):

        # placeholders for inputs
        # 'mean_' is required for tensorboard only
        with tf.name_scope('inputs'):
            self.states = tf.placeholder(shape = [None, self.state_dim], dtype = tf.float32, name = 'states')
            self.actions = tf.placeholder(shape = [None, self.num_actions], dtype = tf.int32, name = 'actions')
            self.state_returns = tf.placeholder(shape = [None, ], dtype = tf.float32, name = 'state_returns')
            self.mean_ = tf.placeholder(tf.float32, name = 'mean')

        # create the network -- 3 fully connected layers
        FC1 = tf.layers.dense(
            inputs = self.states,
            units = 16,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name = 'FC1'
        )

        FC2 = tf.layers.dense(
            inputs = FC1,
            units = 32,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='FC2'
        )

        logits = tf.layers.dense(
            inputs = FC2,
            units = self.num_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='FC3'
        )

        # squash the outputs of the network to between 0 and 1
        # so we get a probability distribution over actions
        self.action_probs = tf.nn.softmax(logits, name = 'action_probs')

        # calculate the loss
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.actions)
            self.loss = tf.reduce_mean(neg_log_prob * self.state_returns)

        #op to train the network to minimize the loss
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

        #op for writing info to tensorboard
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Mean', self.mean_)
        self.write_op = tf.summary.merge_all()

    # function to train the network
    # argument 'latest_episode_wise_rewards' is to be passed so
    # that it can be written to tensorboard
    def train(self, latest_episode_wise_rewards):

        # obtain array containing discounted returns
        discounted_returns = self.normalized_discounted_returns()

        # run the training ops
        # passing inputs -- states, actions and returns for one episode
        _, train_loss = self.sess.run([self.optimizer, self.loss], feed_dict = {
            self.states : np.vstack(np.array(self.episode_states)),
            self.actions : np.vstack(np.array(self.episode_actions)),
            self.state_returns : discounted_returns
        })

        # calculate the mean of the last 100 episodes
        mean = np.mean(latest_episode_wise_rewards)

        # run the op to write to tensorboard --
        # training loss and score for one episode
        summary = self.sess.run(self.write_op, feed_dict = {
            self.states : np.vstack(np.array(self.episode_states)),
            self.actions : np.vstack(np.array(self.episode_actions)),
            self.state_returns : discounted_returns,
            self.mean_ : mean
        })
        self.writer.add_summary(summary, self.episode)
        self.writer.flush()

        # periodically save the model
        if self.episode % SAVE_INTERVAL == 0:
            self.saver.save(self.sess, MODEL_DIR + 'model.ckpt')
            print('Model saved')

        # after training has completed, clear the buffers
        # for the next episode
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        self.episode += 1

    def select_action(self, state):

        # obtain the probability distribution of actions given the current state
        probs = self.sess.run(self.action_probs, feed_dict = {self.states : state.reshape([1, self.state_dim])})
        # sample an action from this distributoion
        return np.random.choice(range(probs.shape[1]), p = probs.ravel())

    # function to restore a session from an existing file
    def restore(self):
        self.saver.restore(self.sess, MODEL_DIR + 'model.ckpt')

# function to write to a .csv file the mean calculated
# after every episode
def write_means(means):
    csv = open(MEANS_CSV_FILE, 'w')
    column_title_row = 'index, score\n'
    csv.write(column_title_row)
    for index in range(len(means)):
        row = str(index) + ', ' + str(means[index]) + '\n'
        csv.write(row)

# function for the entire learning process
def learn():

    # create the gym environment and
    # appropriately seed it for reproducibility
    env = gym.make(ENV_NAME)
    env.seed(RNG_SEED)

    # create the agent with specified hyperparameters
    agent = Agent(env, LEARNING_RATE, GAMMA)

    # array containing the rewards for the latest 100 episodes
    latest_episode_wise_rewards = deque(maxlen = 100)

    # begin training loop
    for episode in range(NUM_LEARN_EPISODES):
        state = env.reset()
        episode_net_reward = 0
        step = 0
        done = False

        # begin episode loop
        while not done:
            action = agent.select_action(state)
            state_next, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            episode_net_reward += reward
            step += 1
            state = state_next

        # print and store net reward for this episode
        print('EPISODE %d' %episode, '\t| REWARD %f' %episode_net_reward, '\t| STEPS %d' %step)
        latest_episode_wise_rewards.append(episode_net_reward)

        # train the network
        agent.train(latest_episode_wise_rewards)

# function to play from a saved model
def play():

    # create the gym environment and
    # appropriately seed it for reproducibility
    env = gym.make(ENV_NAME)
    env.seed(RNG_SEED)

    # create agent with a saved model
    agent = Agent(env)
    agent.restore()

    # array containing the rewards for the latest 100 episodes
    latest_episode_wise_rewards = deque(maxlen = 100)

    # array containing means calculated after
    # every episode after the 100th
    means = []

    # begin play loop
    for episode in range(NUM_PLAY_EPISODES):
        state = env.reset()
        episode_net_reward = 0
        done = False

        # begin episode loop
        while not done:

            # display real-time video if RENDER set to True
            if RENDER: env.render()

            action = agent.select_action(state)
            state_next, reward, done, _ = env.step(action)
            episode_net_reward += reward
            state = state_next

        # print and store net reward for this episode
        print('EPISODE %d' %episode, '\t| REWARD %f' %episode_net_reward)
        latest_episode_wise_rewards.append(episode_net_reward)

        # start calculating the means once we have 100+ episodes
        if episode > 100:
            mean = np.mean(latest_episode_wise_rewards)
            means.append(mean)

    # write the means to the .csv file
    write_means(means)


# if specified directory for model exists
# use it to play in the environment
# else train an agent from scratch
def main():
    if os.path.exists(MODEL_DIR):
        play()
    else:
        learn()

if __name__ == '__main__':
    main()
