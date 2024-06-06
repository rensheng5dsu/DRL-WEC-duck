import re
import numpy as np
import math
import gzip
import utils.properties as properties


DELTA_T = properties.DELTA_T
TIME_INTERVAL = properties.TIME_INTERVAL
TIME_STEP = int(TIME_INTERVAL / DELTA_T)
WRITE_INTERVAL = properties.WRITE_INTERVAL

class Processor:
    def __init__(self):
        self.start_time = 0

    def modify_Dict(self, mode, cur_time, env_path, action=[0]):
        """
        Modify the control dictionary and dynamic mesh dictionary based on the mode and action.

        :param mode: The operation mode, either 'step' or 'reset'.
        :type mode: str
        :param cur_time: The current simulation time.
        :type cur_time: float
        :param env_path: Path to the simulation environment.
        :type env_path: str
        :param action: The action taken at this step.
        :type action: list of float, optional
        """
        cur_time = round(cur_time, 3)

        if mode == 'step':
            self.replace_line(env_path + '/backgroundAndBuoy/system/controlDict', 24, 'endTime ' + str(cur_time) + ';')
            if action < 0:
                for i in range(5):
                    self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 59 + i, '')
            else:
                self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 59, '        OrientationConstraint')
                self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 60, '        {')
                self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 61, '            sixDoFRigidBodyMotionConstraint orientation;')
                self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 62, '            centreOfRotation (0 0.005 -0.05);')
                self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 63, '        }')
        elif mode == 'reset':
            self.start_time = cur_time
            self.replace_line(env_path + '/backgroundAndBuoy/system/controlDict', 24, 'endTime ' + str(cur_time) + ';')
            for i in range(5):
                self.replace_line(env_path + '/backgroundAndBuoy/constant/dynamicMeshDict', 59 + i, '')

    def replace_line(self, file_path, line_num, new_text):
        """
        Replace a specific line in a file with new text.

        :param file_path: The path to the file.
        :type file_path: str
        :param line_num: The line number to replace (1-based index).
        :type line_num: int
        :param new_text: The new text to insert.
        :type new_text: str
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines[line_num - 1] = new_text + '\n'

        with open(file_path, 'w') as f:
            f.writelines(lines)

    def get_omega(self, log_path, time):
        """
        Calculate angular velocity (omega) from log file.

        :param log_path: Path to the log file.
        :type log_path: str
        :param time: The current simulation time.
        :type time: float
        :return: The angular velocity (omega).
        :rtype: float
        """
        pattern = r'    Angular velocity: \([^ ]+ ([^ ]+) [^ ]+\)'
        if time % 1 == 0:
            time = int(time)
        target_line = 'Time = ' + str(time) + '\n'
        with open(log_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if target_line in line:
                try:
                    omegaline = lines[i + 12]
                    result = re.findall(pattern, omegaline)
                    return float(result[0])
                except:
                    print('Unable to find the target number')
                    return float(0)
        print('Target line does not exist')
        return float(0)

    def get_omega_txt(self, file_path, time):
        """
        Retrieve the angular velocity (omega) from a text file based on the given time.

        :param file_path: Path to the directory containing the files.
        :type file_path: str
        :param time: The current simulation time.
        :type time: float
        :return: The angular velocity (omega) corresponding to the given time.
        :rtype: float
        """
        with open(file_path + '/myAngularV.txt', 'r') as file:
            omega = [float(line.strip()) for line in file]
        with open(file_path + '/myTime.txt', 'r') as file:
            T = [float(line.strip()) for line in file]

        T_sum = properties.START
        for i in range(len(T)):
            T_sum += T[i]
            if round(T_sum, 3) == time:
                return omega[i]
        print('Cannot find corresponding omega.')
        return None

    def get_wave_height(self, time, cellid_path, file_path):
        """
        Calculate the wave height at a specific time from the cell data.

        :param time: The current simulation time.
        :type time: float
        :param cellid_path: Path to the cell ID file.
        :type cellid_path: str
        :param file_path: Path to the directory containing the files.
        :type file_path: str
        :return: The calculated wave height.
        :rtype: float
        """
        cellList = np.loadtxt(open(cellid_path, "rb"), delimiter=",", skiprows=1, usecols=[5]).astype(int)
        gridX = properties.GRIDX
        gridY = properties.GRIDY

        if time % 1 == 0:
            time = int(time)
        with gzip.open(file_path + str(time) + "/alpha.water.gz", 'rt') as f:
            alphaWater = f.readlines()[23:-11]

        with gzip.open(file_path + str(time) + "/V.gz", 'r') as f:
            cellVolume = f.readlines()[23:-5]

        surfaceElevation = 0
        for i in cellList:
            cellHeight = float(cellVolume[i]) / (gridX * gridY)
            cellAlphaWater = round(float(alphaWater[i]), 5)
            surfaceElevation += (cellHeight * cellAlphaWater)

        return surfaceElevation

    def cal_angular(self, curtime, filepath, pointID, benchmark_x, benchmark_z):
        """
        Calculate the angular displacement based on the current time and benchmark points.

        :param curtime: The current simulation time.
        :type curtime: float
        :param filepath: Path to the directory containing the files.
        :type filepath: str
        :param pointID: The ID of the point to calculate.
        :type pointID: int
        :param benchmark_x: The x-coordinate of the benchmark point.
        :type benchmark_x: float
        :param benchmark_z: The z-coordinate of the benchmark point.
        :type benchmark_z: float
        :return: The calculated angular displacement.
        :rtype: float
        """
        def angle(x1, y1, x2, y2):
            return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

        curtime = round(curtime, 5)
        curtime = int(curtime) if int(curtime) == curtime else float(curtime)
        pointpath = filepath + str(curtime) + "/polyMesh/points.gz"
        with gzip.open(pointpath, 'r') as f:
            current_context = f.readlines()[pointID + 20]
            x = float(current_context.decode('utf-8').split(' ')[0][1:])
            z = float(current_context.decode('utf-8').split(' ')[2][:-2])
            current_angular = angle(0, -0.05, x, z) - angle(0, -0.05, benchmark_x, benchmark_z)
        return current_angular

    def cal_state_info(self, env_path, cur_time, pointID, benchmark_x, benchmark_z):
        """
        Calculate the state information including wave height, angular displacement, and angular velocity.

        :param env_path: Path to the simulation environment.
        :type env_path: str
        :param cur_time: The current simulation time.
        :type cur_time: float
        :param pointID: The ID of the point to calculate.
        :type pointID: int
        :param benchmark_x: The x-coordinate of the benchmark point.
        :type benchmark_x: float
        :param benchmark_z: The z-coordinate of the benchmark point.
        :type benchmark_z: float
        :return: The state information including wave heights, angular displacement, and angular velocity.
        :rtype: tuple of float
        """
        cur_omega = self.get_omega(env_path + '/backgroundAndBuoy/log.overInterDyMFoam', cur_time)
        cur_wave_height = self.get_wave_height(cur_time, 'cellIDs/id_-0.6875_p1.csv', env_path + '/backgroundAndBuoy/processor1/')
        cur_wave_height1 = self.get_wave_height(cur_time, 'cellIDs/id_-1_p1.csv', env_path + '/backgroundAndBuoy/processor1/')
        cur_wave_height2 = self.get_wave_height(cur_time, 'cellIDs/id_-1.3125_p0.csv', env_path + '/backgroundAndBuoy/processor0/')
        cur_wave_height3 = self.get_wave_height(cur_time, 'cellIDs/id_-1.625_p0.csv', env_path + '/backgroundAndBuoy/processor0/')
        cur_wave_height4 = self.get_wave_height(cur_time, 'cellIDs/id_-1.9375_p0.csv', env_path + '/backgroundAndBuoy/processor0/')
        cur_wave_height5 = self.get_wave_height(cur_time, 'cellIDs/id_-2.25_p0.csv', env_path + '/backgroundAndBuoy/processor0/')
        cur_wave_height6 = self.get_wave_height(cur_time, 'cellIDs/id_-2.5625_p0.csv', env_path + '/backgroundAndBuoy/processor0/')
        cur_wave_height7 = self.get_wave_height(cur_time, 'cellIDs/id_-2.875_p0.csv', env_path + '/backgroundAndBuoy/processor0/')
        cur_angular = self.cal_angular(cur_time, env_path + '/backgroundAndBuoy/processor1/', pointID, benchmark_x, benchmark_z)

        return (round(cur_wave_height, 3), round(cur_wave_height1, 3), round(cur_wave_height2, 3),
                round(cur_wave_height3, 3), round(cur_wave_height4, 3), round(cur_wave_height5, 3),
                round(cur_wave_height6, 3), round(cur_wave_height7, 3), round(cur_angular, 2),
                round(cur_omega, 3))

    def get_reward(self, cur_omega, lasttime, curtime):
        """
        Calculate the reward based on the angular velocity and time difference.

        :param cur_omega: The current angular velocity.
        :type cur_omega: float
        :param lasttime: The previous time step.
        :type lasttime: float
        :param curtime: The current time step.
        :type curtime: float
        :return: The calculated reward.
        :rtype: float
        """
        reward = abs(round((cur_omega ** 2) * round(curtime - lasttime, 6), 6))
        return reward
