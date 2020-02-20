
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       
#include <device_functions.h>

#define SIZE 32
#define TSIZE 1024

using namespace std;


__global__ void addKernel(int *A, int *B)
{
	__shared__ int auxMatrix[SIZE][SIZE];
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int pos = col + row * SIZE;
	//if 0,0
	if (row == 0 && col == 0)
		auxMatrix[row][col] = A[pos] + A[pos + 1] + A[pos + SIZE];
	//if 0,31
	else if (row == 0 && col == 31)
		auxMatrix[row][col] = A[pos] + A[pos - 1] + A[pos + SIZE];
	//if 31,0
	else if (row == 31 && col == 0)
		auxMatrix[row][col] = A[pos] + A[pos + 1] + A[pos - SIZE];
	//if 31,31
	else if (row == 31 && col == 31)
		auxMatrix[row][col] = A[pos] + A[pos - 1] + A[pos - SIZE];
	//if row 0 only
	else if (row == 0)
		auxMatrix[row][col] = A[pos] + A[pos + 1] + A[pos - 1] + A[pos + SIZE];
	//if row 31 only
	else if (row == 31)
		auxMatrix[row][col] = A[pos] + A[pos + 1] + A[pos - 1] + A[pos - SIZE];
	//if col 0 only
	else if (col == 0)
		auxMatrix[row][col] = A[pos] + A[pos + 1] + A[pos - SIZE] + A[pos + SIZE];
	//if col 1 only
	else if (col == 31)
		auxMatrix[row][col] = A[pos] + A[pos - 1] + A[pos - SIZE] + A[pos + SIZE];
	//if central position in the matrix
	else if (row != 0 && col != 0 && row != 31 && col != 31)
		auxMatrix[row][col] = A[pos] + A[pos + 1] + A[pos - 1] + A[pos + SIZE] + A[pos - SIZE];

	__syncthreads();
	//printf("\nA[%d]=%d", pos, A[pos]);
	//printf("\nA[%d]=%d\nauxMatrix[%d][%d]=%d", pos, A[pos], row, col, auxMatrix[row][col]);
	//printf("\nCol: %d , threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d", col, threadIdx.x, blockIdx.x, blockDim.x);
	//printf("\nRow: %d , threadIdx.y: %d, blockIdx.y: %d, blockDim.y: %d", row, threadIdx.y, blockIdx.y, blockDim.y);
	B[pos] = auxMatrix[row][col];
	//printf("\nB[%d]=%d", pos, B[pos]);
}

int main()
{
	int A[SIZE][SIZE], B[SIZE][SIZE];
	srand(time(NULL));
	for (int i = 0; i < SIZE; i++)
	{
		for (int k = 0; k < SIZE; k++)
		{
			A[i][k] = rand() % 99;
		}
	}
    // Add vectors in parallel.
	int *dev_A, *dev_B;
	size_t sharedMem = 64;
	dim3 dimGrid(2, 2);
	dim3 dimBlock(16, 16);
	cudaMalloc((void**)&dev_A, SIZE * SIZE *sizeof(int));
	cudaMalloc((void**)&dev_B, SIZE * SIZE *sizeof(int));
	cudaMemcpy(dev_A, A, SIZE *SIZE *sizeof(int), cudaMemcpyHostToDevice);
	addKernel <<<dimGrid, dimBlock, sharedMem>>> (dev_A, dev_B);
	cudaDeviceSynchronize();
	cudaMemcpy(B, dev_B, 32 * 32 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\n\n--A--");
	for (int i = 0; i < SIZE; i++)
	{
		printf("\n[");
		for (int j = 0; j < SIZE; j++)
		{
			if (j != 31)
				if (A[i][j] / 10 == 0)
					printf("00%d-", A[i][j]);
				else if (A[i][j] / 100 == 0)
					printf("0%d-", A[i][j]);
				else
					printf("%d-", A[i][j]);
			else
				if (A[i][j] / 10 == 0)
					printf("00%d", A[i][j]);
				else if (A[i][j] / 100 == 0)
					printf("0%d", A[i][j]);
				else
					printf("%d", A[i][j]);
		}
		printf("]");
	}
	printf("\n\n--B--");
	for (int i = 0; i < SIZE; i++)
	{
		printf("\n[");
		for (int j = 0; j < SIZE; j++)
		{
			if(j!=31)
				if (B[i][j] / 10 == 0)
					printf("00%d-", B[i][j]);
				else if(B[i][j]/100 == 0)
					printf("0%d-", B[i][j]);
				else
					printf("%d-", B[i][j]);
			else
				if (B[i][j] / 10 == 0)
					printf("00%d", B[i][j]);
				else if (B[i][j] / 100 == 0)
					printf("0%d", B[i][j]);
				else
					printf("%d", B[i][j]);
		}
		printf("]");
	}
	// cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_A);
	cudaFree(dev_B);

    return 0;
}
