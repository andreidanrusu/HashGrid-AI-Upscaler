# 2D Image AI upscaler using multi resolution hash grids

## Description
Initially inteded as a building block for a pipeline similar to instant NGP, it became an image reconstructor that has the option to upscale images.

## Setup

### HashGrid2D 
__When initialized 3 parameters can be passed: `hash_size`, `cell_size`, `dimensions`.__  
  
`hash_size` dictates the number of entries a given hash grid can have. The total number is 2^`hash_size`  
  
`cell_size` represents how much space a given entry occupies in a 2D space. If cell_size = 1 then it occupies 1 pixel. This can be adjusted to capture finer and coarser details from an image.  
  
`dimensions` is the size of each feature vector from the table e.g. for 2D a feature vector can be [0.15912, -0.2332]  

The `hash` function prevents collision while keeping the entries as sparse as possible.  

`batch_lookup` essentially finds entries in the grid based on a set of coordinates.

### MultiResHG2D 
  
The given `layout` in the constructor determines the shape of the hash grid. It takes a list of tuples of the form [(int, float, int)]. 

__Layout__  
The layout which I found to be most stable (i.e. lightweight and pretty accurate) is this [(14, 8.0, 4), (16, 2.0, 8), (18, 0.5, 8)].  
It is also the default option when running the __Trainer__.  
It consists of:  
1. A 4D hash grid that learns 8x8 squares of pixels. It captures broader details while keeping memory usage low.  
2. A 8D grid that captures 2x2 squares that captures finer details.  
3. A 8D grid that blends colors between pixels, essentially learning how to sharpen the image while preserving the structure.


Dependng on usage, other layouts might be more efficient. For example, while training with the gradient picture `gradient.jpeg`, I found a (14, 4.0, 4) to be most efficient.


__Adjustments__  
During training I noticed that some levels would dominate over the others so I used softmax to stabilize them. 


TODO: Finish doc
