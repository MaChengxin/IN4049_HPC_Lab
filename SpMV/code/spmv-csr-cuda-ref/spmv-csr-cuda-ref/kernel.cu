// parallel SPMV CSR
// compute y = y + Ax
// Assume all the input element of vector y[i] and x[i] are "i", and length(y) = length(x) = col[A]
//
// G. Fu, Dec 31, 2015
//

#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

// used in csrmul_kernel
__device__ float multiply_row(int rowsize,
	int *Aj,      // column indices for row
	float *Av,    // non-zero entries for row
	float *x)     // the RHS vector
{
	float sum = 0;
	for (int column = 0; column < rowsize; ++column)
		sum += Av[column] * x[Aj[column]];
	return sum;
}

// used in main
__global__ void csrmul_kernel(int *Ap, int *Aj, float *Av, int num_rows_A, float *x, float *y)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;  //parallel by the row number
	if (row < num_rows_A)
	{
		int row_begin = Ap[row];
		int row_end = Ap[row + 1];
		y[row] = y[row] + multiply_row(row_end - row_begin, Aj + row_begin, Av + row_begin, x);
	}
}

int main(void)
{
	const int num_Ap = 5;
	const int num_Av = 9;
	const int num_rows_A = 4;
	const int num_rows_x = 4;
	const int num_rows_y = 4;

	int Ap_host[] = { 0, 2, 4, 7, 9 };
	int Aj_host[] = { 0, 1, 1, 2, 0, 2, 3, 1, 3 };
	float Av_host[] = { 1.0, 7.0, 2.0, 8.0, 5.0, 3.0, 9.0 , 6.0, 4.0 };
	float x_host[] = { 1.0, 2.0, 3.0, 4.0 };
	float y_host[] = { 1.0, 2.0, 3.0, 4.0 };

	int *Ap;
	int *Aj;
	float *Av;
	float *x;
	float *y;

	int size_Ap = num_Ap * sizeof(int);
	int size_Aj = num_Av * sizeof(int);
	float size_Av = num_Av * sizeof(float);
	float size_x = num_rows_x * sizeof(float);
	float size_y = num_rows_y * sizeof(float);

	for (int i = 0; i <= num_rows_y - 1; i++)
		printf("input y(%i) = %f\n", i, y_host[i]);

	cudaMalloc((void**)&Ap, size_Ap);
	cudaMalloc((void**)&Aj, size_Aj);
	cudaMalloc((void**)&Av, size_Av);
	cudaMalloc((void**)&x, size_x);
	cudaMalloc((void**)&y, size_y);

	cudaMemcpy(Ap, Ap_host, size_Ap, cudaMemcpyHostToDevice);
	cudaMemcpy(Aj, Aj_host, size_Aj, cudaMemcpyHostToDevice);
	cudaMemcpy(Av, Av_host, size_Av, cudaMemcpyHostToDevice);
	cudaMemcpy(x, x_host, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(y, y_host, size_y, cudaMemcpyHostToDevice);

	unsigned int blocksize = 128;
	unsigned int nblocks = (num_rows_A + blocksize - 1) / blocksize;

	csrmul_kernel << < nblocks, blocksize >> > (Ap, Aj, Av, num_rows_A, x, y);

	cudaMemcpy(y_host, y, size_y, cudaMemcpyDeviceToHost);

	cudaFree(y);
	cudaFree(x);
	cudaFree(Av);
	cudaFree(Aj);
	cudaFree(Ap);

	for (int i = 0; i <= num_rows_y - 1; i++)
		printf("output y(%i) = %f\n", i, y_host[i]);

	return EXIT_SUCCESS;
}
