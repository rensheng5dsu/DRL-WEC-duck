import utils.properties as properties
import subprocess
import os
import numpy as np
import data_process
import utils.tools as tools
import matplotlib.pyplot as plt
import scipy.io as sio

foam_env = os.environ.copy()
foam_env['WM_PROJECT_DIR'] = '/home/user/OpenFOAM-V1812/OpenFOAM-v1812'
foam_env['MPI_HOME'] = '/usr/local/openmpi/bin:/usr/local/openmpi/lib:/usr/local/openmpi/share/man'


class Simulator:
    """
    A simulator for wave energy converter (WEC) control using OpenFOAM.

    Attributes:
        OF_file_path (str): Path to the OpenFOAM directory.
        processor (Processor): An instance of the data_process.Processor class.
        start_time (float): The start time of the simulation.
        end_time (float): The end time of the simulation.
        initinal_angel (float): The initial angle of the WEC.
        initinal_omega (float): The initial angular velocity of the WEC.
        pointID (int): The processor point ID for data processing.
        benchmark_x (float): The x-coordinate of the benchmark point.
        benchmark_z (float): The z-coordinate of the benchmark point.
        last_time (float): The last recorded simulation time.
        last_state (np.ndarray): The last recorded state of the simulation.
    """

    def __init__(self, OF_path):
        """
        Initialize the Simulator.

        :param OF_path: The file path to the OpenFOAM directory.
        :type OF_path: str
        """
        self.OF_file_path = OF_path
        self.processor = data_process.Processor()
        self.start_time = properties.START
        self.end_time = properties.END
        self.initinal_angel = properties.INI_ANGEL
        self.initinal_omega = properties.INI_OMEGA
        self.pointID = properties.PROCESSOR_POINT_ID
        self.benchmark_x = properties.BENCHMARK_X
        self.benchmark_z = properties.BENCHMARK_Z
        self.last_time = self.start_time
        self.last_state = None

    def reset(self):
        """
        Reset the simulation environment.

        This method deletes part of the simulation environment and resets the
        state to the initial conditions.

        :return: The initial state of the simulation environment.
        :rtype: np.ndarray
        """
        # Delete part of the simulation environment
        cmd1 = './deletePart'
        subprocess.run(cmd1, shell=True, cwd=self.OF_file_path + '/backgroundAndBuoy', stdout=subprocess.DEVNULL)

        # Modify the dictionary to reset state
        self.processor.modify_Dict('reset', self.start_time, self.OF_file_path)

        # Define the initial state
        orig_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -17.1])
        self.last_state = orig_state

        return orig_state

    def step(self, action, cur_time):
        """
        Execute a single time step in the environment.
        :param action: The control action for this time step.
        :type action: int or float
        :param cur_time: The current simulation time.
        :type cur_time: float

        :return:
            - cur_states (np.ndarray): The current state of the system.
            - reward (float): The reward obtained from the action.
            - done (bool): Whether the simulation has reached its end.
            - a_cur (float): The current angular value.
            - omega_cur (float): The current omega value.
        """
        # Modify the dictionary with the current step information
        self.processor.modify_Dict('step', cur_time, self.OF_file_path, action)

        # Commands to execute the simulation step and post-process the results
        cmd1 = './next-run-noReconstruct'
        cmd2 = 'postProcess -func "writeCellVolumes" -time ' + str(round(cur_time, 2))

        # Execute the commands
        subprocess.run(cmd1, shell=True, cwd=self.OF_file_path, stdout=subprocess.DEVNULL)
        subprocess.run(cmd2, shell=True,
                       cwd=self.OF_file_path + '/backgroundAndBuoy/processor0/',
                       stdout=subprocess.DEVNULL)
        subprocess.run(cmd2, shell=True,
                       cwd=self.OF_file_path + '/backgroundAndBuoy/processor1/',
                       stdout=subprocess.DEVNULL)

        # Calculate the current angular value
        a_cur = self.processor.cal_angular(cur_time, self.OF_file_path + '/backgroundAndBuoy/processor1/',
                                           self.pointID, self.benchmark_x, self.benchmark_z)

        # Determine the current omega value based on the action
        if action > 0:
            omega_cur = 0.0
        else:
            omega_cur = self.processor.get_omega_txt(self.OF_file_path + '/backgroundAndBuoy', cur_time)

        # Calculate the current state information
        cur_wave_height, cur_wave_height1, cur_wave_height2, cur_wave_height3, cur_wave_height4, cur_wave_height5, \
            cur_wave_height6, cur_wave_height7, cur_angular, cur_omega = \
            self.processor.cal_state_info(self.OF_file_path, cur_time, self.pointID, self.benchmark_x, self.benchmark_z)

        # Aggregate the state information into a numpy array
        cur_states = np.array([cur_wave_height, cur_wave_height1, cur_wave_height2, cur_wave_height3, cur_wave_height4,
                               cur_wave_height5, cur_wave_height6, cur_wave_height7, cur_angular])

        # Calculate the reward for the current state and action
        reward = self.processor.get_reward(cur_omega, self.last_time, cur_time)

        # Update the last state and time
        self.last_state = cur_states
        self.last_time = cur_time

        # Determine if the simulation is done
        if cur_time >= self.end_time:
            done = True
        else:
            done = False

        return cur_states, reward, done, a_cur, omega_cur

    def plt_ep(self, ep, actions, rotations, omegas, ep_reward, id):
        """
        Plot and save the episode results.

        :param ep: The current episode number.
        :type ep: int
        :param actions: Array of actions taken during the episode.
        :type actions: np.ndarray
        :param rotations: Array of rotations during the episode.
        :type rotations: np.ndarray
        :param omegas: Array of omega values during the episode.
        :type omegas: np.ndarray
        :param ep_reward: Total reward accumulated during the episode.
        :type ep_reward: float
        :param id: Identifier for the episode plot.
        :type id: int
        """
        # Define paths
        path_duck = self.OF_file_path + '/backgroundAndBuoy/processor1/'
        path_rotation = self.OF_file_path + '/natural_rotation.txt'
        cellid = np.loadtxt(open("cellIDs/id_-0.6875_p1.csv", "rb"), delimiter=",", skiprows=1, usecols=[5]).astype(int)

        # Load data
        t_noControl = np.loadtxt(path_rotation)[:, 0]
        r_noControl = np.loadtxt(path_rotation)[:, 1]
        t_action = actions[:, 0]
        r_action = actions[:, 1]
        t_wave, h_wave = tools.run_Cell(cellid, path_duck, t_action)

        # Correct data for plotting
        tools.correct(t_action, 1, 0)
        tools.correct(r_action, 1, -7.86)
        tools.correct(rotations, 1, 16.83)
        tools.correct(r_noControl, 1, 0)
        tools.correct(h_wave, 1, -0.5)

        # Plot the data
        plt.figure(figsize=(20, 12))
        plt.title(f"Total Work: {ep_reward:.6f}")
        plt.plot(t_action[1:], rotations[1:], color='green', label='Actively controlled rotation', linewidth=1)
        plt.plot(t_action, omegas, color='black', label='Actively controlled omega', linewidth=1)
        plt.scatter(t_action, omegas, s=4, c='m', alpha=0.5, label='omega')
        plt.plot(t_noControl, r_noControl, color='red', label='rotation', linewidth=1, linestyle='-.')
        plt.plot(t_wave, h_wave, color='blue', label='wave_duckfront', linewidth=1, linestyle=':')

        # Save data to .mat file
        data_dict = {'tRotate': t_action[1:],
                     'rotate': rotations[1:],
                     'tOmega': t_action,
                     'omega': omegas,
                     'tWave': t_wave,
                     'wave': h_wave}
        sio.savemat(properties.OF_FILE_PATH + 'mats/' + str(ep) + '-' + str(id) + '.mat', data_dict)

        # Finalize plot
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('height')
        plt.savefig(properties.OF_FILE_PATH + 'pictures/' + str(ep) + '-' + str(id) + '.png')
        plt.clf()

    def cal_rotations_omegas(self, start, end):
        """
        Calculate rotations and omegas over a time range.

        :param start: The start time for the calculations.
        :type start: float
        :param end: The end time for the calculations.
        :type end: float

        :return:
            - time_series (list): List of time points.
            - rotations (list): List of calculated rotations at each time point.
            - omegas (list): List of calculated omega values at each time point.
        """
        time_series = []
        rotations = []
        omegas = []

        # Iterate over the time range
        for i in range(int(round((end - start) / properties.TIME_INTERVAL))):
            cur_time = round(start + (i + 1) * properties.TIME_INTERVAL, 2)
            a_cur = self.processor.cal_angular(cur_time, self.OF_file_path + '/backgroundAndBuoy/processor1/',
                                               self.pointID, self.benchmark_x, self.benchmark_z)
            omega_cur = self.processor.get_omega_txt(self.OF_file_path + '/backgroundAndBuoy', cur_time)
            time_series.append(cur_time)
            rotations.append(a_cur)
            omegas.append(omega_cur)

        return time_series, rotations, omegas
