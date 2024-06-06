import numpy as np
import gzip
import properties
import math


def sample(time, cellList, filepath):
    """
    Calculate the free surface elevation.

    :param time: Simulation time
    :param cellList: List of cell IDs within a gauge (fine grid)
    :param filepath: Path to the simulation data files
    :return: Free surface elevation monitored by the gauge at the given simulation time
    """
    # Define grid dimensions
    gridX = properties.GRIDX
    gridY = properties.GRIDY

    # Adjust time format for file path
    if time % 1 == 0:
        time = int(time)

    # Read alpha.water data
    with gzip.open(f"{filepath}{time}/alpha.water.gz", 'rt') as f:
        alphaWater = f.readlines()[23:-11]

    # Read cell volume data
    with gzip.open(f"{filepath}{time}/V.gz", 'r') as f:
        cellVolume = f.readlines()[23:-5]

    # Calculate free surface elevation
    surfaceElevation = 0
    for i in cellList:
        cellHeight = float(cellVolume[i]) / (gridX * gridY)
        cellAlphaWater = round(float(alphaWater[i]), 5)
        surfaceElevation += (cellHeight * cellAlphaWater)

    return surfaceElevation


def run_Cell(cellid, filepath, times):
    """
    Calculate the free surface elevation over a series of time steps.

    :param cellid: List of cell IDs
    :param filepath: Path to the simulation data files
    :param times: List of time steps
    :return: List of time steps and corresponding free surface elevations
    """
    height = []
    for t in times:
        height.append(sample(round(t, 2), cellid, filepath))
    return times, height


def correct(data_list, zoom, translate):
    """
    Correct the xy data by applying zoom and translation.

    :param data_list: List of data points
    :param zoom: Zoom factor
    :param translate: Translation value
    """
    for i in range(len(data_list)):
        data_list[i] = data_list[i] * zoom + translate


def angle_change(x1, y1, x2, y2, x3, y3):
    """
    Calculate the change in angle from fixed and rotated points.

    :param x1, y1: Coordinates of the fixed point
    :param x2, y2: Coordinates of the first rotated point
    :param x3, y3: Coordinates of the second rotated point
    :return: Change in angle
    """

    def angle(x1, y1, x2, y2):
        """Calculate the angle between two points."""
        return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

    return angle(x1, y1, x3, y3) - angle(x1, y1, x2, y2)
