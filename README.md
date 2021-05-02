# Cuda_image_quantization

In this repo we will implement image quantization for grayscale
images. This method reduces the number of colors u sed in an image
Below
you can see a pair of images. On the left, the original image, and on
the right, the image after quantization to 4 colors
(https://www.flickr.com/photos/crsan/4379532286)

![image](https://user-images.githubusercontent.com/37774604/116829540-edf65300-abac-11eb-8f85-b46674a6d00d.png)


The algorithm we will implement works by first creating a histogram of the
gray levels in the image, then using the histogram to create a map w hich
maps each value of these levels to a new value, and finally, using this map to
create the new image.
We will implement quantization on MxN grayscale images, each represented
as a unsigned char array of length M*N, with values in the range [0,255],
w here 0 means black, and 255 means white.


We
perform quantization into L gray levels as follows:
1. Create a histogram h an array of length 256. h[v] is the number of pixels
which have the value v.
2. Create the cumulative distribution function CDF from the histogram.
CDF[v]=h[0]+h[1]+...+h[v]
3. Create a map m from old gray level to new gray level. m[v] is the new
value of pixels which originally had the value v.
m is computed as follows:

![image](https://user-images.githubusercontent.com/37774604/116829059-6cea8c00-abab-11eb-8c71-2a8f730fde44.png =20x20)

4. Compute the new image For each pixel 𝑖
new[𝑖]=𝑚[original[𝑖]]

## Compilation & Execution

```bash
make image_quantization
./a.out
```

The repository includes the following files:

`image_quantization.cu`     : GPU implementation of the algorithm that is used to check the results.

`image_quantization-cpu.cu` : CPU implementation of the algorithm that is used to check the results.

`main.cu`: test harness that runs your algorithm on random images and compares the result against the CPU implementation a bove, as
well as measure s performance.

`image.cu`: test program that runs the CPU implementation against an image file and produces an output image file, for your curiosity by running:
./image <imagefile>
        
`cat.jpg`: cat shown above.

`Makefile`: Allows building the exercise ex1 and the graphical test application image using make ex1 and make image respectively.

## bulk synchronous version:  
Now we will feed the GPU with all the tasks at once, a version of the kernel which supports being invoked with an array of
images and mulitple threablocks, where the block index determines which image each
threadblock will handle. Alternatively, modify the kernel from (to support multiple threadblocks).

## Test Results


### GPU run time and the throughput
![image](https://user-images.githubusercontent.com/37774604/116829598-3a419300-abad-11eb-8fe2-9ba3af2c5254.png)

### bulk synchronous version
![image](https://user-images.githubusercontent.com/37774604/116829607-462d5500-abad-11eb-8c03-d3e38a67663f.png)


## Environment
        Nvidia GeForce RTX 2080
        Intel® Core™ i7-4700HQ CPU
        Cuda 10.2, V10.2.89

