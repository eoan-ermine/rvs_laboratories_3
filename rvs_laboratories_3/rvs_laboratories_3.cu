#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Место для вставки кода
__global__ void grayscale(float *in, float *out, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    float r = in[3*i];
    float g = in[3*i + 1];
    float b = in[3*i + 2];

    out[i] = (0.21f * r + 0.71f * g + 0.07f * b);
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* чтение входных аргументов */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // В данной лабораторной значение равно 3
  imageChannels = wbImage_getChannels(inputImage);

  //  Так как изображение монохромное, оно содержит только 1 канал
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  wbCheck(cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float)));
  wbCheck(cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  
  auto inputLength = imageWidth * imageHeight;
  int blockSize = 256;
  int gridSize = (inputLength + blockSize - 1) / blockSize;
  grayscale<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData, inputLength);
  wbCheck(cudaDeviceSynchronize());
  wbCheck(cudaGetLastError());
  //@@ Место для вставки кода

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  wbCheck(cudaFree(deviceInputImageData));
  wbCheck(cudaFree(deviceOutputImageData));

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}