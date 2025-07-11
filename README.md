# CNS 2025 NEST GPU software showcase
Efficient simulation of large-scale spiking neuronal networks is important for neuroscientific research, and both the simulation speed and the time it takes to instantiate the network in computer memory are key factors. In recent years, hardware acceleration through highly parallel GPUs has become increasingly popular. NEST GPU is a GPU-based simulator under the NEST Initiative written in CUDA-C++ that demonstrates high simulation speeds with models of various network sizes on single-GPU and multi-GPU systems [1,2,3].
Using a single NVIDIA RTX4090 GPU we have simulated networks on the magnitude of 80 thousand neurons and 200 million synapses with a real time factor of 0.4; and using 12000 NVIDIA A100 GPUs on the LEONARDO cluster we have managed to simulate networks on the magnitude of 3.3 billion neurons and 37 trillion synapses with a real time factor of 20.
In this showcase, we will demonstrate the capabilities of the GPU simulator and present our roadmap to integrate this technology into the ecosystem of the CPU-based simulator NEST [4].
For this, we will focus on three aspects of the simulation across model scales, namely network construction speed, state propagation speed, and energy efficiency.
Furthermore, we will present our efforts to statistically validate our simulation results against those of NEST (CPU) using established network models.
You can follow our progress through our [GitHub](https://github.com/nest/nest-gpu) page.
Materials for this showcase can be found in [Zenodo](https://doi.org/10.5281/zenodo.15754814).

1. Golosio et al. Front. Comput. Neurosci. 15:627620, 2021.
2. Tiddia et al. Front. Neuroinform. 16:883333, 2022.
3. Golosio et al. Appl. Sci. 13, 9598, 2023.
4. Graber, S., et al. NEST 3.8 (3.8). Zenodo. 10.5281/zenodo.12624784 


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
* **figures/**
  * [qr_code.png](figures/qr_code.png): QR Code pointing to showcase material upload in Zenodo
  * [brunel_network.png](figures/brunel_network.png): Network sketch for Brunel network
  * [ring_topology_activity.png](figures/ring_topology_activity.png): Output activity of simulation of part 3
  * [leonardo_ring_benchmark_net_con.png](figures/leonardo_ring_benchmark_net_con.png): Network construction performance of ring network benchmark
  * [leonardo_ring_benchmark_rft.png](figures/leonardo_ring_benchmark_rtf.png): Real-Time factor of ring network benchmark
  * [leonardo_ring_benchmark_time_init.png](figures/leonardo_ring_benchmark_time_init.png): Initialization time performance of ring network benchmark
  * [leonardo_ring_benchmark_time_create.png](figures/leonardo_ring_benchmark_time_create.png): Node creation time performance of ring network benchmark
  * [leonardo_ring_benchmark_time_connect.png](figures/leonardo_ring_benchmark_time_connect.png): Node connection time performance of ring network benchmark
  * [leonardo_ring_benchmark_time_calibrate.png](figures/leonardo_ring_benchmark_time_calibrate.png): Calibration time performance of ring network benchmark
  * [leonardo_ring_benchmark_time_presim.png](figures/leonardo_ring_benchmark_time_presim.png): Pre-simulation time performance of ring network benchmark
  * [leonardo_ring_benchmark_time_sim.png](figures/leonardo_ring_benchmark_time_sim.png): Simulation time performance of ring network benchmark
  * [leonardo_ring_benchmark_time_gather.png](figures/leonardo_ring_benchmark_time_gather.png): Data gathering time performance of ring network benchmark
* **data/**
  * [simulation_data.json](data/simulation_data.json): Simulation output of part 3
  * [timer_data.json](data/timer_data.json): Ring network benchmarking data performed in LEONARDO Booster
* [CNS_2025_NEST_GPU_Showcase.pdf](CNS_2025_NEST_GPU_Showcase.pdf): Showcase slides
* [nest_gpu_CNS_showcase.ipynb](nest_gpu_CNS_showcase.ipynb): Jupyter notebook for parts 1 & 2
  * A Google Colab version of the interactive Jupyter notebook is available [here](https://colab.research.google.com/drive/1FNGmYn4dWMBskJBDY_2lE5yuaugIshl9?usp=sharing)
* [ring_topology_example.py](ring_topology_example.py): Python script for part 3
* [plot_timer_data.py](plot_timer_data.py): Python script to create benchmarking data plots


## Software requirements
Libraries were used with Python 3.11.9
* NEST GPU [mpi_comm branch](https://github.com/nest/nest-gpu/tree/nest-gpu-2.0-mpi-comm)
  * See installation instructions at: https://nest-gpu.readthedocs.io/en/latest/installation/index.html
* [mpi4py](https://pypi.org/project/mpi4py/): tested with version 4.1.0
* [matplotlib](https://pypi.org/project/matplotlib/): tested with version 3.10.3
* [seaborn](https://pypi.org/project/seaborn/): tested with version 0.13.2


## Contact
* Luca Sergi, Department of Physics, University of Cagliary, Italy, Istituto Nazionale di Fisica Nucleare, Sezione di Cagliari, Italy, lsergi@dsf.unica.it
* José Villamar, Institute for Advanced Simulation (IAS-6), Jülich, Germany, j.villamar@fz-juelich.de


## License
GPL 3.0 [license](LICENSE)
