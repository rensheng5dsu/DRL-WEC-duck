import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
from tensorboardX import SummaryWriter
import duckenv
import utils.properties as properties
import queue

# Hyper-parameters
BATCH_SIZE = properties.BATCH_SIZE
LR = properties.LR
GAMMA = properties.GAMMA
EPISILO = properties.EPISILO
EPISILO_DECLINE_RATE = properties.EPISILO_DECLINE_RATE
MEMORY_CAPACITY = properties.MEMORY_CAPACITY
LEARN_CAPACITY = properties.LEARN_CAPACITY
Q_NETWORK_ITERATION = properties.Q_NETWORK_ITERATION
MAX_TIME = properties.END - properties.START
PER_STEP_TIME = properties.TIME_INTERVAL

# Environment setup
env = duckenv.duckEnv(0)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


class Net(nn.Module):
    """
    Neural network for the DQN.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN:
    """
    DQN agent.
    """

    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, random_state, episilon):
        """
        Choose an action based on the current state.

        :param state: The current state.
        :param random_state: Random state for action sampling.
        :param episilon: Exploration rate.
        :return: Chosen action.
        """
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if self.memory_counter > LEARN_CAPACITY and np.random.uniform() <= episilon:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = random_state.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        """
        Store transition in memory.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state.
        """
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """
        Learn from stored transitions.
        """
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, ep_count, path):
        """
        Save the model parameters.

        :param ep_count: Episode count.
        :param path: Path to save the model.
        """
        torch.save({
            'eval_state_dict': self.eval_net.state_dict(),
            'target_state_dict': self.target_net.state_dict()
        }, properties.OF_FILE_PATH + path + str(ep_count) + '.pth')

    def load_model(self, path):
        """
        Load the model parameters.

        :param path: Path to the saved model.
        """
        checkpoint = torch.load(path)
        self.eval_net.load_state_dict(checkpoint['eval_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])


def worker(worker_id, action_queue, state_queue):
    """
    Worker function for multiprocessing.

    :param worker_id: Worker ID.
    :param action_queue: Queue to receive actions.
    :param state_queue: Queue to send state.
    """

    def normalize_state(state):
        return np.round((state - min_states) / (max_states - min_states), 3)

    env = duckenv.duckEnv(worker_id)
    reset_tag = False
    ep_reward = 0
    initial_state = env.reset()
    state_queue.put((worker_id, initial_state, 0, False, ep_reward))
    state = [0, 0, 0, 0, 0]
    max_states = np.array([0.6] * 8 + [-15])
    min_states = np.array([0.4] * 8 + [-18])

    while True:
        try:
            action, ep_num = action_queue.get(timeout=1)
            if action == -1:
                break
            if reset_tag:
                env.reset()
                ep_reward = 0
                reset_tag = False
                next_state, reward, done = initial_state, 0, False
                state = [0, 0, 0, 0, 0]
            else:
                next_state, reward, done, _ = env.step(action, ep_num)
                path = properties.OF_FILE_PATH + "exps/"
                next_state = normalize_state(next_state)
                with open(path + "experiences" + str(worker_id) + ".txt", "a") as f:
                    f.write(f"{ep_num}: {state},{action},{reward},{next_state},{done}\n")
                state = next_state
                ep_reward += reward

            if done:
                reset_tag = True
            state_queue.put((worker_id, next_state, reward, done, ep_reward))

        except queue.Empty:
            continue


def main():
    """
    Main function to train the DQN.
    """
    global EPISILO, EPISILO_DECLINE_RATE
    dqn = DQN()
    writer = SummaryWriter(properties.OF_FILE_PATH + 'logs')
    episodes = properties.EPISODES
    num_workers = properties.NUM_WORKERS
    ep_count = 0
    best_epR = 0
    action_queues = [mp.Queue() for _ in range(num_workers)]
    random_states = [np.random.RandomState(seed) for seed in range(num_workers)]
    state_queue = mp.Queue()
    stop_tag = []
    start_from_second_tag = {i: True for i in range(1, num_workers + 1)}
    states = dict()
    actions = dict()

    processes = [mp.Process(target=worker, args=(i + 1, action_queues[i], state_queue)) for i in range(num_workers)]
    for p in processes:
        p.start()

    while True:
        try:
            worker_id, next_state, reward, done, ep_reward = state_queue.get(timeout=1)

            if not start_from_second_tag[worker_id]:
                if worker_id <= 15:
                    dqn.store_transition(states[worker_id], actions[worker_id], reward, next_state)
                    if dqn.memory_counter > LEARN_CAPACITY:
                        dqn.learn()
            else:
                start_from_second_tag[worker_id] = False

            action = dqn.choose_action(next_state, random_states[worker_id - 1], EPISILO)
            actions[worker_id] = action
            if done:
                start_from_second_tag[worker_id] = True
                ep_count += 1
                if ep_count < 200:
                    EPISILO = min(0.99, EPISILO + EPISILO_DECLINE_RATE)
                else:
                    EPISILO = 0.99
                if ep_reward > best_epR:
                    best_epR = ep_reward
                    dqn.save_model(ep_count, 'model/best/')
                dqn.save_model(ep_count, 'model/')
                writer.add_scalar('train/triwave', ep_reward, ep_count)
            states[worker_id] = next_state
            if ep_count >= episodes:
                action_queues[worker_id - 1].put((-1, episodes))
                stop_tag.append(worker_id)
            else:
                action_queues[worker_id - 1].put((action, ep_count))
            if len(stop_tag) == num_workers:
                break
        except queue.Empty:
            pass

    for p in processes:
        p.join()
    writer.close()


if __name__ == '__main__':
    main()
