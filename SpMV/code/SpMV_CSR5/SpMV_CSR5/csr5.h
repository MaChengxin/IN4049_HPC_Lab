/*
CSR5-Formation
By Jianbing Jin   2016-Jan-6
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEBUG 0
#define True 1
#define False 0

#define max(a,b) ((a)>(b)?a:b) // equal to the "OR operation"

// for computing the w and Sigma
#define w 4   //w is decided by the hardware of the 
int Sigma = 4; // Sigma is set as 4 initially
int p; //the tile number
void Sigma_computing();
// the csr5-parameters
int *tile_ptr;
float ***tile_val;
int ***tile_col_idx;

//the tile descrip
int ***tile_bit_flag;
int **tile_y_offset;
int **tile_seg_offset;
int **tile_empty_offset;
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
void Sigma_computing();
void CSR5_Transformation();
void Write_Grid();
void Clean_Up();

void tile_dexcription();
void tile_val_computing();
void tile_ptr_computing();
void Debug(char *mesg, int terminate);
