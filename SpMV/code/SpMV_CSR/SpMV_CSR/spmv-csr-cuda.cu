/*
Parallel SPMV CSR
Compute y = y + Ax

G. Fu, Dec 31, 2015
Modified by Jianbing Jin
*/

#include "spmv-csr-cuda.h"

int main(void)
{
	Read_Matrix_A_CSR_info();

	Construct_Y_X();

	int *row_ptr_dev;
	int *col_idx_dev;
	float *val_dev;
	float *x_dev;
	float *y_dev;

	int size_row_ptr = (num_row_in_mat + 1) * sizeof(int);
	int size_col_idx = num_none_zero_in_mat * sizeof(int);
	float size_val = num_none_zero_in_mat * sizeof(float);
	float size_x = num_col_in_mat * sizeof(float);
	float size_y = num_col_in_mat * sizeof(float);

	cudaMalloc((void**)&row_ptr_dev, size_row_ptr);
	cudaMalloc((void**)&col_idx_dev, size_col_idx);
	cudaMalloc((void**)&val_dev, size_val);
	cudaMalloc((void**)&x_dev, size_x);
	cudaMalloc((void**)&y_dev, size_y);

	cudaMemcpy(row_ptr_dev, row_ptr, size_row_ptr, cudaMemcpyHostToDevice);
	cudaMemcpy(col_idx_dev, col_idx, size_col_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(val_dev, val, size_val, cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(y_dev, y, size_y, cudaMemcpyHostToDevice);

	unsigned int block_size = 128;
	unsigned int num_blocks = ceil((double)num_row_in_mat / block_size);
	Debug("Calculating on CUDA...", 0);
	csrmul_kernel << < num_blocks, block_size >> > (row_ptr_dev, col_idx_dev, val_dev, num_row_in_mat, x_dev, y_dev);
	Debug("Calculation finished.", 0);

	cudaMemcpy(y, y_dev, size_y, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_col_in_mat; i++)
		printf("output y[%i] = %f\n", i, y[i]);

	cudaFree(row_ptr_dev);
	cudaFree(col_idx_dev);
	cudaFree(val_dev);
	cudaFree(x_dev);
	cudaFree(y_dev);

	free(row_ptr);
	free(col_idx);
	free(val);
	free(x);
	free(y);

	return EXIT_SUCCESS;
}

void Read_Matrix_A_CSR_info()
{
	Debug("Reading Matrix A...", 0);

	FILE *f;
	f = fopen("csr_matrix.dat", "r");
	if (f == NULL)
		Debug("Error opening csr_matrix.dat", 1);

	fscanf(f, "(row*column, none_zero_num): (%i*%i, %i)\n", &num_row_in_mat, &num_col_in_mat, &num_none_zero_in_mat);
	printf("The input file's fisrt row is: %i %i %i\n", num_row_in_mat, num_col_in_mat, num_none_zero_in_mat);

	row_ptr = (int *)malloc((num_row_in_mat + 1) * sizeof(int));
	if (row_ptr == NULL)
		Debug("Read_Matrix_A : malloc(row_ptr) failed", 1);
	col_idx = (int *)malloc(num_none_zero_in_mat * sizeof(int));
	if (col_idx == NULL)
		Debug("Read_Matrix_A : malloc(col_idx) failed", 1);
	val = (float *)malloc(num_none_zero_in_mat * sizeof(float));
	if (val == NULL)
		Debug("Read_Matrix_A : malloc(val) failed", 1);

	fscanf(f, "row_ptr:\n");
	for (int i = 0; i < num_row_in_mat + 1; i++)
	{
		fscanf(f, "%i\n", &row_ptr[i]);
		printf("row_ptr[%i]: %i\n", i, row_ptr[i]);
	}

	fscanf(f, "col_idx and val:\n");
	for (int i = 0; i < num_none_zero_in_mat; i++)
	{
		fscanf(f, "%i %f", &col_idx[i], &val[i]);
		printf("col_idx[%i] and val[%i]: %i and %f\n", i, i, col_idx[i], val[i]);
	}

	Debug("Matrix A (CSR format) read.", 0);
}

void Construct_Y_X()
{
	Debug("Constructing y and x...", 0);

	y = (float *)malloc(num_col_in_mat * sizeof(float));
	if (y == NULL)
		Debug("Construct_Y_X : malloc(y) failed", 1);

	x = (float *)malloc(num_col_in_mat * sizeof(float));
	if (x == NULL)
		Debug("Construct_Y_X : malloc(x) failed", 1);

	for (int i = 0; i < num_col_in_mat; i++)
	{
		x[i] = i + 1.0;
		y[i] = i + 1.0;
		printf("x[%i] and input y[%i]: %f and %f\n", i, i, x[i], y[i]);
	}

	Debug("y and x constructed.", 0);

}

void Debug(char *mesg, int terminate)
{
	if (DEBUG || terminate)
		printf("%s\n", mesg);
	if (terminate)
		exit(1);
}

__device__ float multiply_row(int row_size, int *idx, float *val, float *x)
{
	float sum = 0.0;
	for (int i = 0; i < row_size; ++i)
		sum += val[i] * x[idx[i]];
	return sum;
}

__global__ void csrmul_kernel(int *row_ptr, int *col_idx, float *val, int num_row_in_mat, float *x, float *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_row_in_mat)
		y[i] = y[i] + multiply_row(row_ptr[i + 1] - row_ptr[i], col_idx + row_ptr[i], val + row_ptr[i], x);
}
