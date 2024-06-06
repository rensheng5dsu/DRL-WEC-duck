# DRL-WEC-duck

This is the official implementation of the paper "**Optimization of Latching Control for Duck Wave Energy Converter Based on Deep Reinforcement Learning**" .

## Abstract
In the field of wave energy extraction, employing active control strategies amplifies the Wave Energy Converter's (WEC) response to wave motion. In this regard, a numerical simulation study was conducted on the interaction between Duck WEC and waves under discrete latching control, proposing an optimized latching duration strategy based on Deep Reinforcement Learning (DRL). The study utilized an improved Edinburgh Duck model to establish a numerical wave flume (NWF), verifying the motion of waves and the device. A DRL and Computational Fluid Dynamics (CFD) coupled framework was developed to implement the latching control strategy. In the limited trichromatic wave testing conditions provided in this paper, the latching control time is optimized using a DRL agent, and its capture efficiency is compared with two non-predictive latching control strategies. The DRL agent control showed a 13.4% improvement relative to reference threshold control, with less than 6.3% differences compared to optimal threshold latching results. Improvement of approximately 5.6% was observed compared to optimal constant latching control. For the first time in nonlinear fluid dynamics numerical simulation, this study demonstrates the efficacy of model-free reinforcement learning with discrete latching actions, establishing an environment-driven control strategy, and having the potential to become a general strategy for WEC latching control.

![image-20240606185849780](C:\Users\13600\AppData\Roaming\Typora\typora-user-images\image-20240606185849780.png)




## How to Use

### Requirements

use requirements.txt to install the required packages
```bash
pip install -r requirements.txt
```

### Prepare OpenFOAM case
The location of the OpenFOAM case needs to be specified in OF_FILE_PATH in utils.properties. The training results will also be generated in this location.

### Train the Model

The model can be trained using the script in 'main.py'. Experiment parameters can be adjusted within the utils.properties to meet the training process to specific needs.
```bash
python main.py
```


## Contact
We welcome contributions and collaborations. If you are interested in this project or have any questions, please feel free to contact us:

Email: sshw@cug.edu.cn