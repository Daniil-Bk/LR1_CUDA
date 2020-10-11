#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cstdlib>

cudaError_t cudaMul(int *c, const int *a, const int *b, unsigned int size);
void multiplyOnHost(int* a, int row1, int col1, int* b, int row2, int col2, int* c);
int* get_random_array(int* arr, int n, int from, int to);
bool compareArr(int* arr1, int* arr2, int size);

const int BLOCK_SIZE = 10; // BLOCK SIZE

__global__ void multiplyOnDevice(int* c, int* a, int* b, int n)
{
    int blockX = blockIdx.x, blockY = blockIdx.y;
    int threadX = threadIdx.x, threadY = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * blockY;
    int aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * blockX;
    int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    int sum = 0;
    for (int indexA = aBegin, indexB = bBegin; indexA <= aEnd; indexA += aStep, indexB += bStep)
    {
        __shared__ int as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int bs[BLOCK_SIZE][BLOCK_SIZE];
        as[threadY][threadX] = a[indexA + n * threadY + threadX];
        bs[threadY][threadX] = b[indexB + n * threadY + threadX];
        __syncthreads(); // Убедимся, что подматрицы полностью загружены
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[threadY][k] * bs[k][threadX];
        __syncthreads(); // Убедимся, что подматрицы никому больше не нужны
    }
    
    c[n * BLOCK_SIZE * blockY + BLOCK_SIZE * blockX + n * threadY + threadX] = sum;
}

int main()
{
    int N = 2000;
    const int arraySize = N*N;
    int* a = new int[arraySize];
    int* b = new int[arraySize];
    get_random_array(a, arraySize, -100, 100);
    get_random_array(b, arraySize, -100, 100);
  
    int* cH = new int[arraySize];
    int* cD = new int[arraySize];

    clock_t beginH = clock();
    multiplyOnHost(a, N, N, b, N, N, cH);
    double hostTime = double(clock() - beginH) * 1000 / CLOCKS_PER_SEC;
    
    cudaError_t cudaStatus = cudaMul(cD, a, b, N);
   
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    bool compare = compareArr(cH, cD, arraySize);

    if (compare) {
        printf("true\n");
        printf("Time CPU: %f\n", hostTime);
    }
    else {
        printf("falce");
    }
    free(a);
    free(b);
    free(cD);
    free(cH);
    return 0;
}

cudaError_t cudaMul(int *c, const int *a, const int *b, unsigned int N)
{
    const int size = N * N;
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    clock_t beginD = clock();
    multiplyOnDevice <<<dim3(N / BLOCK_SIZE, N / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(dev_c, dev_a, dev_b, N);
    cudaThreadSynchronize();
    double deviceTime = double(clock() - beginD) * 1000 / CLOCKS_PER_SEC;
    printf("Time GPU: %f\n", deviceTime);
   
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyOnDevice launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyOnDevice!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

int rand_between(const int from, const int to) {
    if (to == from)
        return to;
    if (to < from)
        return rand_between(to, from);
    return from + rand() % (to - from + 1);
}

int* get_random_array(int* arr, int n, int from, int to) {
    
    for (int i = 0; i < n; ++i) {
        arr[i] = rand_between(from, to);
    }
    return arr;
}

bool compareArr(int* arr1, int* arr2, int size) {
    for (int i = 0; i < size; ++i)
    {
        if (arr1[i] != arr2[i])
            return false;
    }

    return true;
}

void multiplyOnHost(int* a, int row1, int col1, int* b, int row2, int col2, int* c)
{
   
    int size = row1 * col2;
   
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            int sum = 0;
            for (int k = 0; k < col1; k++)
                sum = sum + a[i * col1 + k] * b[k * col2 + j];
            c[i * col2 + j] = sum;
        }
    }

}