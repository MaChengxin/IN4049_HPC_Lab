/*
Parallel SPMV CSR
Compute y = y + Ax

G. Fu, Dec 31, 2015
Modified by Jianbing Jin
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define DEBUG 1

/* global variables */
int num_row_in_mat;
int num_col_in_mat; // same as the size of x and y
int num_none_zero_in_mat;

int *row_ptr;
int *col_idx;
float *val;
float *x;
float *y;

void Read_Matrix_A_CSR_info();
void Construct_Y_X();
void Debug(char *mesg, int terminate);
__device__ float multiply_row(int rows_size, int *idx, float *val, float *x);
__global__ void csrmul_kernel(int *A_ptr, int *A_idx, float *A_val, int num_rows_A, float *x, float *y);
