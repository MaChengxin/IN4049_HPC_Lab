/*
CSR5-Formation
By Jianbing Jin   2016-Jan-6
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG 0
#define True 1
#define False 0

#define max(a,b) ((a)>(b)?a:b)

// for computing the w and sigma
#define omega 4   //omega is decided by the hardware
int sigma = 4; // sigma is set as 4 initially
int p; //the tile number

// csr5-parameters
int *tile_ptr;
float ***tile_val;
int ***tile_col_idx;

// tile description
int ***tile_bit_flag;
int **tile_y_offset;
int **tile_seg_offset;
int **tile_empty_offset;

// input variable 
int row_num;
int col_num;
int none_zero_num;
int *row;
int *col;
float *val_input;

//output
int *row_ptr;
int *col_idx;
float *val_output;

float **val_Matrix;
int **row_Matrix;
int **col_Matrix;

void Read_Matrix();
void CSR_Transformation();
void Compute_Sigma();
void CSR5_Transformation();
void Write_Matrix_Info();
void Clean_Up();

void Compute_Tile_Ptr();
void Compute_Tile_Val();
void Compute_Tile_Desc();

void Debug(char *mesg, int terminate);
