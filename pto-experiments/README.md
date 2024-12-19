# PTO Experiments
This folder includes all experiments using the Program Trace Optimization code. The basic idea is to compare the performance of different discrete search algorithms (Random Search, Hill Climbers, Genetic Algorithms, Particle Swarm Optimization) on some of the ARC tasks. The hypothesis is that the fitness function is not great.

## Setup
First, install the PTO python package.
```
pip install git+https://github.com/Program-Trace-Optimisation/PTO.git
```
Then, take the `re-arc` repository by Michael Hodel, and clone it inside the `../../local` directory.
```
git clone https://github.com/michaelhodel/re-arc.git ../../local/re-arc
```
Unzip the .zip file `../../local/re-arc/arc_original.zip` inside a subfolder `../../local/re-arc/arc_original/`. Now you are ready!