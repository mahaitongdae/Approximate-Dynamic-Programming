# Vehicle Tracking Control

- Code demo for Chpater 8, Reinforcement Learning and Control.

- Methods: Approximate Dynamic Programming, Model Predictive Control

<div align=center>
<img src="road.png" width = 50%/>
</div>

## Requirements

[PyTorch](https://pytorch.org/get-started/previous-versions/)  1.4.0

[CasADi](https://web.casadi.org/get/)


## Getting Started

- To train an agent, follow the example code in `main.py` and tune the parameters. Change `METHODS` variable for adjusting the methods to compare in simulation stage.
- Simulations will automatically executed after the training is finished. To separately start a simulation from a trained results and compare the performance between ADP and MPC, run `Simulation.py`. Change `LOG_DIR` variable to set the loaded results.

## Directory Structure

```
Approximate-Dynamic-Programming
│  main.py - Main script
│  plot.py - To plot comparison between ADP and MPC
│  train.py - To execute PEV and PIM
│  Dynamics.py - Vehicle model
│  Network.py - Network structure
│  Solver.py - Solvers for MPC using CasADi
│  Config.py - Configurations about training and vehicle model
│  readme.md
│  requirements.txt
│
├─Results_dir - store trained results
│     
└─Simulation_dir - store simulation data and plots

```
## Related Books and Papers
[Reinforcement Learning and Control. Tsinghua University
Lecture Notes, 2020.](http://www.idlab-tsinghua.com/thulab/labweb/publications.html?typeId=3&_types)

[CasADi: a software framework for nonlinear optimization and optimal control](https://link.springer.com/article/10.1007/s12532-018-0139-4)



