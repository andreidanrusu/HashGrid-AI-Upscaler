# NeRF-like reconstruction of 2D images

A prototype I made to learn Instant-NGP pipelines.
This consists of:

 - How and why hash grids are used
 - What is hash collision and how is it mitigated
 - Memory benefits and tradeoffs between various data structures (voxel grids vs hash grids vs 
 - How hash grid parameters influence the details they capture (hash size, cell size and dimensions)
 - What are feature vectors and why are they used
 - How can multiple hash grids of different sizes be combined into a multi resolution one
 - How images are vectorized to improve the performance of the MLP
 - MLP network design for NeRF + hash grids
 - Specific application of various loss functions (MSE, L1, AMP etc.)
 - What is interpolation and different variations (bilinear, bicubic, NN etc.)

The learning processes and code portions were facilitated by ChatGPT.

Moving to a C++/CUDA implementation due to performance limitations of PyTorch in real time rendering and training.
