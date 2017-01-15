// parallel SPMV CSR
// compute y = y + Ax
// Assume all the input element of vector y[i] and x[i] are "i", and length(y) = length(x) = col[A]
//
// G. Fu, Dec 31, 2015
//
// Modified by Jianbing Jin

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//#include <malloc.h>




#define DEBUG 0

/* global variables */
int row_num;
int col_num;
int none_zero_num;

int *row_ptr_input;
int *col_idx_input;
float *val_input;
float *x_input;
float *y_input;

//the input parameters of matrix A, vector x and vector y//






// compute multiply_row   used in csrmul_kernel
__device__ float multiply_row(int rowsize,
                              int *col_idx,      // column indices for row
                              float *val,    // non-zero entries for row
                              float *x)     // the RHS vector
{
	float sum = 0;
	for (int column = 0; column < rowsize; ++column)
		sum += val[column] * x[col_idx[column]];
	return sum;
}

// compute CSR format, kernel for Matrix-vector multiplication
__global__ void csrmul_kernel(int *row_ptr, int *col_idx, float *val, int num_rows_A,
                              float *x, float *y)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x ;  //parallel by the row number
	if (row < num_rows_A)
	{
		int row_begin = row_ptr[row];
		int row_end = row_ptr[row + 1];
		y[row] = y[row] + multiply_row(row_end - row_begin, col_idx + row_begin, val + row_begin, x);
	}
}

void Read_Matrix_A();
void Construction_Y_X();
void Debug(char *mesg, int terminate);
void Clean_Up();



int main(void)
{
	int *row_ptr;
	int *col_idx;
	float *val;
	float *x;
	float *y;


	printf("hehe\n");

	Read_Matrix_A();

	Construction_Y_X();

	int size_row_ptr = (row_num + 1) * sizeof(*row_ptr);
	int size_col_idx = none_zero_num * sizeof(*col_idx);
	float size_val = none_zero_num * sizeof(*val);
	float size_x = col_num * sizeof(float);
	float size_y = col_num * sizeof(float);

	printf("1\n");
	printf("size_row_ptr, size_col_idx, size_val, size_x, size_y are %i %i %f %f %f\n", size_row_ptr, size_col_idx, size_val, size_x, size_y);


	for ( int ii = 0 ; ii <= col_num - 1; ii++ )
	{
		printf("input y(%i) = %f\n", ii, y_input[ii]);
	}


	cudaMalloc( (void**)&row_ptr, size_row_ptr );
	cudaMalloc( (void**)&col_idx, size_col_idx );
	cudaMalloc( (void**)&val, size_val );
	cudaMalloc( (void**)&x, size_x );
	cudaMalloc( (void**)&y, size_y );

	cudaMemcpy( row_ptr, row_ptr_input, size_row_ptr, cudaMemcpyHostToDevice );
	cudaMemcpy( col_idx, col_idx_input, size_col_idx, cudaMemcpyHostToDevice );
	cudaMemcpy( val, val_input, size_val, cudaMemcpyHostToDevice );
	cudaMemcpy( x, x_input, size_x, cudaMemcpyHostToDevice );
	cudaMemcpy( y, y_input, size_y, cudaMemcpyHostToDevice );


	unsigned int blocksize = 128; //
	unsigned int nblocks = (row_num + blocksize - 1) / blocksize;
	csrmul_kernel <<< nblocks, blocksize>>>(row_ptr, col_idx, val, row_num, x, y);

	cudaMemcpy(y_input, y, size_y, cudaMemcpyDeviceToHost);

	// cleanup memory
	cudaFree(y);
	cudaFree(x);
	cudaFree(val);
	cudaFree(col_idx);
	cudaFree(row_ptr);

	for ( int ii = 0 ; ii <= col_num - 1; ii++ )
	{
		printf("output y(%i) = %f\n", ii, y_input[ii]);
	}


	Clean_Up(); //clean up the memory for the input parameters
	return EXIT_SUCCESS;

}

void Read_Matrix_A()
{
	FILE *f;
	int i;

	Debug("Read_Matrix_A", 0);

	f = fopen("csr_matrix.dat", "r");
	if (f == NULL)
		Debug("Error opening csr_matrix.dat", 1);

	fscanf(f, "csr_matrix_formate(row*column, none_zero_num): %i*%i, %i\n", &row_num, &col_num, &none_zero_num);
	printf("the inputfile's fisrt row is %i %i %i\n", row_num, col_num, none_zero_num);

	//row_ptr_input = (int *)malloc(sizeof(int));
	//Debug("Read_Matrix_A : malloc(row_ptr_input) failed", 1);

	row_ptr_input = (int *)malloc((row_num + 1) * sizeof(int));
	// Debug("Read_Matrix_A : malloc(row_ptr) failed", 1);
	col_idx_input = (int *)malloc(none_zero_num * sizeof(int));
	// Debug("Read_Matrix_A : malloc(col_idx) failed", 1);
	val_input = (float *)malloc(none_zero_num * sizeof(float));
	//Debug("Read_Matrix_A : malloc(val) failed", 1);

	fscanf(f, "row_ptr:\n");
	for (i = 0; i < row_num + 1; i++)
	{
		fscanf(f, "%i\n", &row_ptr_input[i]);
		printf("row_ptr_input[%i] is %i\n", i, row_ptr_input[i]);
	}

	fscanf(f, "col_idx and val:\n");
	for (i = 0; i < none_zero_num; i++)
	{
		fscanf(f, "%i %f", &col_idx_input[i], &val_input[i]);
		printf("col_idx and val_input[%i] is %i %f\n", i, col_idx_input[i], val_input[i]);
	}
}

void Construction_Y_X()
{

	int i;

	//allocate memory for the vector x_input and y_input
	y_input = (float *)malloc(none_zero_num * sizeof(float));
	// Debug("Construction_Y_X : malloc(y_input) failed", 1);
	x_input = (float *)malloc(none_zero_num * sizeof(float));
	// Debug("Construction_Y_X : malloc(row_ptr) failed", 1);

	for (i = 0; i < col_num; i++)
	{
		x_input[i] = i + 1.0;
		y_input[i] = i + 1.0;
		printf("x[%i] and y[] is %f %f\n", i, x_input[i], y_input[i]);
	}

}
void Debug(char *mesg, int terminate)
{
	if (DEBUG || terminate)
		printf("%s\n", mesg);
	if (terminate)
		exit(1);
}
void Clean_Up()
{
	Debug("Clean_Up", 0);

	free(row_ptr_input);
	free(col_idx_input);
	free(val_input);
	free(x_input);
	free(y_input);
}
