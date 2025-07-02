# CNS 2025 NEST GPU software showcase

A Google Colab version of the interactive Jupyter notebook is available [here](https://colab.research.google.com/drive/1FNGmYn4dWMBskJBDY_2lE5yuaugIshl9?usp=sharing).

## Summary
This is a short showcase for the NEST GPU software.
A rough guideline of the presentation:

* Introduction: What is NEST and what is NEST GPU? (10 min)
  * spiking neural network simulation technology
  * NEST as a simulation tool for point-neuron network models from laptops to supercomputers
  * NEST GPU as the GPU backend for NEST
    * CUDA backend + MPI for scaling, similar Python interface
    * part of the NEST Initiative, and the aim is to merge with NEST
* Part 1: Modeling networks with NEST GPU (20 min)
  * Step by step of a Brunel network using Jupyter notebook
* Part 2: Analyzing spiking data from NEST GPU (20 min)
  * How to get and store spiking data from NEST GPU
  * Basics on how to process it
    * transforming the spiking data to spike trains
    * using elephant to get firing rates, cv isi, correlations
    * some simple plot
* Part 3: Scaling networks with NEST GPU (20 min)
  * Minimal introduction to GPU computing hardware and distributed computing paradigm
  * Short introduction to Python interface functions required to coordinate MPI network construction
  * Multi-GPU Brunel network with feed forward ring topology as an example use case
* Conclusion and outlook (10 min)
  * Related works
    * Validation
    * Performance analysis
  * Future plans
* Q&A / buffer time (10 min)

## Files included here
* [CNS_2025_NEST_GPU_Showcase.pdf](CNS_2025_NEST_GPU_Showcase.pdf): Showcase slides
* [brunel_network.png](brunel_network.png): Network sketch for Brunel network
* [nest_gpu_CNS_showcase.ipynb](nest_gpu_CNS_showcase.ipynb): Jupyter notebook for parts 1 & 2
* [ring_topology_example.py](ring_topology_example.py): Python script for part 3
* [ring_topology_activity.png](ring_topology_activity.png): Output activity of simulation of part 3
* [simulation_data.json](simulation_data.json): Simulation output of part 3

## Software requirements
Libraries were used with Python 3.11.9
* NEST GPU [mpi_comm branch](https://github.com/nest/nest-gpu/tree/nest-gpu-2.0-mpi-comm)
  * See installation instructions at: https://nest-gpu.readthedocs.io/en/latest/installation/index.html
* [mpi4py](https://pypi.org/project/mpi4py/): tested with version 4.1.0
* [matplotlib](https://pypi.org/project/matplotlib/): tested with version 3.10.3

## Contact
Luca Sergi, Department of Physics, University of Cagliary, Italy, Istituto Nazionale di Fisica Nucleare, Sezione di Cagliari, Italy, lsergi@dsf.unica.it
José Villamar, Institute for Advanced Simulation (IAS-6), Jülich, Germany, j.villamar@fz-juelich.de

## License
GPL 3.0 [license](LICENSE)
