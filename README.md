# 2D Image AI Upscaler Using Multi-Resolution Hash Grids

## Description  
Initially intended as a building block for a pipeline similar to Instant NGP, it became an image reconstructor that has the option to upscale images.

## Setup

### HashGrid2D  
__When initialized, 3 parameters can be passed: `hash_size`, `cell_size`, and `dimensions`.__  
  
`hash_size` dictates the number of entries a given hash grid can have. The total number is 2^`hash_size`.  
  
`cell_size` represents how much space a given entry occupies in a 2D space. If `cell_size = 1`, then it occupies 1 pixel. This can be adjusted to capture finer and coarser details from an image.  
  
`dimensions` is the size of each feature vector from the table, e.g., for 2D a feature vector can be [0.1529, 0.2336].

The `hash` function prevents collisions while keeping the entries as sparse as possible.  

`batch_lookup` essentially finds entries in the grid based on a set of coordinates.

### MultiResHG2D  

The given `layout` in the constructor determines the shape of the hash grid. It takes a list of tuples of the form `[(int, float, int)]`.

__Layout__  
The layout that I found to be most stable (i.e., lightweight and fairly accurate) is `[(14, 8.0, 4), (16, 2.0, 8), (18, 0.5, 8)]`.  
It is also the default option when running the __Trainer__.  
It consists of:  
1. A 4D hash grid that learns 8x8 squares of pixels. It captures broader details while keeping memory usage low.  
2. An 8D grid that captures 2x2 squares, allowing it to capture finer details.  
3. An 8D grid that blends colors between pixels, essentially learning how to sharpen the image while preserving structure.

Depending on usage, other layouts might be more efficient. For example, while training with the gradient picture `gradient.jpeg`, I found `(14, 4.0, 4)` to be the most efficient.

__Adjustments__  
During training, I noticed that some levels would dominate over the others, so I used softmax to stabilize them.

### MLP

Consists of a basic 3-layer MLP with as many inputs as the sum of dimensions in the hash grid layout, 64 neurons in the hidden layer, and 4 in the output layer (initially designed  
to produce density as an output for 3D images but dropped it later on due to computational limitations).  
  
### Trainer  

This is where the trainer is assembled. The constructor takes the `path` to a given file, `batch_size` (how many hash grid entries are computed before the loss is calculated),  
and the layout described above. Here the grid, MLP, learning rate, and optimizer are initialized. The image is also split into batches so that it can be fed into the MLP.  

The `train(epoch)` method essentially feeds each batch into the MLP, computes the loss, and uses the optimizer to update the weights of the MLP.  
It is important to note that if a CUDA device is detected, it will attempt to use float16 precision, which does not perform best with RTX 20 GPUs or lower.  
For that, I recommend float32 to ensure accuracy with the model.  
  
The `reconstruct_image(samples, save_path)` essentially takes values from the hash grid and approximates them into (R, G, B) by denormalizing them from [0, 1] to [0, 255]. The image is  
stretched based on the sample size and reconstructed using bicubic interpolation. If a `path` is passed, then it is saved; otherwise, it's displayed using pyplot.

### CUI  

This is a basic implementation of a console interface with 4 options. It walks you through all the necessary initializations for the `Trainer` and gives the option to train on an image for a given number of epochs. The image can be displayed or saved.

## Results  

### Images  

Below is a list of images used for training.  
  
1. ![Image 1](main/data/images/girl_128.png)  
   128x128  
2. ![Image 2](main/data/images/waves.jpg)  
   258x258  
3. ![Image 3](main/data/images/empire_state.jpg)  
   500x500  
4. ![Image 4](main/data/images/colors.jpg)  
   700x600

### Performance  

Below is the image generated after running the trainer with the following settings:  
`Layout`  [(14, 8.0, 4), (16, 2.0, 8), (18, 0.5, 8), (18, 0.125, 4)]  

`Epochs`  500  
`Sampling`  4x  

1. `Completion time`: 4 minutes 6.76 seconds  
   `Batch size`: 128  
   ![Sampled Image 1](main/data/images/girl_128_final_500E_4S_14H-8.0C-4D_16H-2.0C-8D_18H-0.5C-8D_18H-0.125C-4D.jpeg)  
    
2. `Completion time`: 8 minutes 52.87 seconds  
   `Batch size`: 128  
   ![Sampled Image 2](main/data/images/waves_final_500E_4S_14H-8.0C-4D_16H-2.0C-8D_18H-0.5C-8D_18H-0.125C-4D.jpeg)  
    
3. `Completion time`: 15 minutes 43.81 seconds  
   `Batch size`: 512  
   ![Sampled Image 3](main/data/images/empire_state_final_500E_4S_14H-8.0C-4D_16H-2.0C-8D_18H-0.5C-8D_18H-0.125C-4D.jpeg)  
  
4. `Completion time`: 13 minutes 32.08 seconds  
   `Batch size`: 1024  
   ![Sampled Image 4](main/data/images/colors_final_500E_4S_14H-8.0C-4D_16H-2.0C-8D_18H-0.5C-8D_18H-0.125C-4D.jpeg)

## Discussion  

The results were presented as such to showcase the `limitations` of this trainer as image size increases. On top of that, as it stands, the trainer as a product is essentially a glorified bicubic upscaler that introduces artifacts in images.  

The purpose of this project was to get acquainted with how a 2D image reconstruction pipeline is implemented and the practical aspects of adjusting MLP parameters. Another important learning objective was understanding how multi-layer hash grids are designed to capture a wide variety of features from images while keeping memory usage as low as possible. These completed learning objectives allowed me to kickstart the implementation of the same algorithm in C++/CUDA while being conscientious of how design choices impact performance and image quality.
