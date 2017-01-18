/*
CSR-Formation by Jianbing Jin, 2016-Jan-6
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#define DEBUG 1

void Read_Matrix();
void CSR_Transformation();
void Write_Matrix();
void Clean_Up();
void Debug(char *mesg, int terminate);

// input variable
int row_size;
int col_size;
int none_zero_num;

int *row;
int *col;
float *val_in;

//output
int *row_ptr;
int *col_idx;
float *val_out;
