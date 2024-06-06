from gym import spaces
import my_simulator
import numpy as np

import utils.properties as properties


class duckEnv():
    def __init__(self, processor_id):
        self.simulator = my_simulator.Simulator(properties.OF_FILE_PATH + 'Tstep0.01-'
                                                + str(processor_id))
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, -90]),
                                            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 0]),
                                            shape=(9,), dtype=np.float32)

        self.rotations = [properties.INI_ANGEL]
        self.action_time = [properties.START]
        self.omegas = [properties.INI_OMEGA]
        self.action_single = [0]
        self.current_time = properties.START

        self.processor_id = processor_id
        self.ep_reward = 0
        self.ep_reward_work = 0
        self.last_state = None
        np.random.seed(processor_id)

    def reset(self):
        self.rotations = [properties.INI_ANGEL]
        self.action_time = [properties.START]
        self.omegas = [properties.INI_OMEGA]
        self.action_single = [0]
        self.current_time = properties.START
        self.last_state = None
        self.ep_reward = 0
        self.ep_reward_work = 0
        self.simulator.reset()
        for _ in range(100):
            self.current_time += properties.TIME_INTERVAL
            self.current_time = round(self.current_time, 2)
            next_state, step_reward, done, info_r, info_omega = self.simulator.step(-1, self.current_time)
            new_times, new_rotations, new_omegas = self.simulator.cal_rotations_omegas(
                round(self.current_time - properties.TIME_INTERVAL, 2),
                self.current_time)
            self.rotations.extend(new_rotations)
            self.action_single.extend([0] * len(new_times))
            self.omegas.extend(new_omegas)
            self.action_time.extend(new_times)

            if abs(self.omegas[-1]) <= 0.025:  # latching control of speed threshold
                break

        return next_state

    def step(self, action, ep_num):
        reward = 0
        for _ in range(1, action + 2):  # execute latching control
            for _ in range(5):  # Avoid differences in time steps of 0.01 time step and 0.05 action
                temp_lasttime = self.current_time
                self.current_time += properties.TIME_INTERVAL
                self.current_time = round(self.current_time, 2)
                next_state, step_reward, done, info_r, info_omega = self.simulator.step(1, self.current_time)
                reward += step_reward
                new_times, new_rotations, new_omegas = self.cal_rotations_omegas(temp_lasttime,
                                                                                 self.current_time)
                self.rotations.extend(new_rotations)
                self.omegas.extend(new_omegas)
                self.action_time.extend(new_times)
                self.action_single.extend([1] * len(new_times))
        self.last_state = next_state
        while True:  # release duck until arriving  into speed threshold
            self.current_time += properties.TIME_INTERVAL
            self.current_time = round(self.current_time, 2)
            next_state, step_reward, done, info_r, info_omega = self.simulator.step(-1, self.current_time)
            reward += step_reward
            new_times, new_rotations, new_omegas = self.cal_rotations_omegas(
                round(self.current_time - properties.TIME_INTERVAL, 2),
                self.current_time)
            self.rotations.extend(new_rotations)
            self.action_single.extend([0])
            self.omegas.extend(new_omegas)
            self.action_time.extend(new_times * len(new_times))
            if abs(self.omegas[-1]) <= 0.025:
                break
        self.ep_reward_work += reward
        self.ep_reward += reward
        info = 0
        action_t_a = np.column_stack((np.array(self.action_time), np.array(self.action_single)))
        if done:
            self.plt_ep(ep_num,
                        np.array(action_t_a),
                        np.array(self.rotations),
                        np.array(self.omegas),
                        round(self.ep_reward_work, 3))
        return next_state, reward, done, info

    def plt_ep(self, ep, actions, rotations, omegas, ep_reward):
        self.simulator.plt_ep(ep, actions, rotations, omegas, ep_reward, self.processor_id)

    def cal_rotations_omegas(self, start, end):
        return self.simulator.cal_rotations_omegas(start, end)
