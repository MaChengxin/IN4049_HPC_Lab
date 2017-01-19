/*
Parallel SPMV CSR
Compute y = y + Ax
G. Fu, Dec 31, 2015
Modified by C. Ma
*/

#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__device__ float multiply_row(int row_size, int *idx, float *val, float *x)
{
	float sum = 0.0;
	for (int i = 0; i < row_size; ++i)
		sum += val[i] * x[idx[i]];
	return sum;
}

__global__ void csrmul_kernel(int *A_ptr, int *A_idx, float *A_val, int num_rows_A, float *x, float *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_rows_A)
		y[i] = y[i] + multiply_row(A_ptr[i + 1] - A_ptr[i], A_idx + A_ptr[i], A_val + A_ptr[i], x);
}

int main(void)
{
	int A_ptr[] = { 0, 2, 4, 7, 9 };
	int A_idx[] = { 0, 1, 1, 2, 0, 2, 3, 1, 3 };
	float A_val[] = { 1.0, 7.0, 2.0, 8.0, 5.0, 3.0, 9.0 , 6.0, 4.0 };
	float x[] = { 1.0, 2.0, 3.0, 4.0 };
	float y[] = { 1.0, 2.0, 3.0, 4.0 };

	int num_rows_A = 4;
	int num_rows_y = sizeof(y) / sizeof(y[0]);

	for (int i = 0; i < num_rows_y; i++)
		printf("input y(%i) = %f\n", i, y[i]);

	int *A_ptr_dev;
	int *A_idx_dev;
	float *A_val_dev;
	float *x_dev;
	float *y_dev;

	int size_A_ptr = sizeof(A_ptr);
	int size_A_idx = sizeof(A_idx);
	float size_A_val = sizeof(A_val);
	float size_x = sizeof(x);
	float size_y = sizeof(y);

	cudaMalloc((void**)&A_ptr_dev, size_A_ptr);
	cudaMalloc((void**)&A_idx_dev, size_A_idx);
	cudaMalloc((void**)&A_val_dev, size_A_val);
	cudaMalloc((void**)&x_dev, size_x);
	cudaMalloc((void**)&y_dev, size_y);

	cudaMemcpy(A_ptr_dev, A_ptr, size_A_ptr, cudaMemcpyHostToDevice);
	cudaMemcpy(A_idx_dev, A_idx, size_A_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(A_val_dev, A_val, size_A_val, cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(y_dev, y, size_y, cudaMemcpyHostToDevice);

	unsigned int block_size = 128;
	unsigned int num_blocks = (num_rows_A + block_size - 1) / block_size;

	csrmul_kernel << < num_blocks, block_size >> > (A_ptr_dev, A_idx_dev, A_val_dev, num_rows_A, x_dev, y_dev);

	cudaMemcpy(y, y_dev, size_y, cudaMemcpyDeviceToHost);

	cudaFree(A_ptr_dev);
	cudaFree(A_idx_dev);
	cudaFree(A_val_dev);
	cudaFree(x_dev);
	cudaFree(y_dev);

	for (int i = 0; i < num_rows_y; i++)
		printf("output y(%i) = %f\n", i, y[i]);

	return EXIT_SUCCESS;
}
