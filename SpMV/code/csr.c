/*CSR-Formation  
 *   By Jianbing Jin   2016-Jan-6
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEBUG 0


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

float *val;
clock_t ticks;			/* number of systemticks */
int timer_on = 0;		/* is timer running? */

void Read_Matrix();
void CSR_Transformation();
double Do_Step(int parity);
void Solve();

void Write_Grid();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();


void start_timer()
{
  if (!timer_on)
  {
    ticks = clock();
    timer_on = 1;
  }
}

void resume_timer()
{
  if (!timer_on)
  {
    ticks = clock() - ticks;
    timer_on = 1;
  }
}

void stop_timer()
{
  if (timer_on)
  {
    ticks = clock() - ticks;
    timer_on = 0;
  }
}

void print_timer()
{
  if (timer_on)
  {
    stop_timer();
    printf("Elapsed processortime: %14.6f s\n", ticks * (1.0 / CLOCKS_PER_SEC));
    resume_timer();
  }
  else
    printf("Elapsed processortime: %14.6f s\n", ticks * (1.0 / CLOCKS_PER_SEC));
}

void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("%s\n", mesg);
  if (terminate)
    exit(1);
}

void Read_Matrix()
{
  int x, s, i = 0;
  FILE *f;
  int source_x;
  int source_y;
  float source_val;
  Debug("Read_Matrix", 0);
  
  f = fopen("full-matrix-input2.dat", "r");
  if (f == NULL)
    Debug("Error opening full-matrix-input.dat", 1);
  
  // read the row_num column_num and none_zero_num of the input matrix
  fscanf(f, "%i %i %i\n", &row_num, &col_num, &none_zero_num);
  printf("the fisrt row is %i %i %i\n", row_num, col_num, none_zero_num);
   /* allocate memory for the input matrix */
  if ((row = malloc(none_zero_num * sizeof(*row))) == NULL)
    Debug("Read_Matrix : malloc(row) failed", 1);
  if ((col = malloc(none_zero_num * sizeof(*col))) == NULL)
    Debug("Read_Matrix : malloc(col) failed", 1);
  if ((val_input = malloc(none_zero_num * sizeof(*val_input))) == NULL)
    Debug("Read_Matrix : malloc(val_input) failed", 1);

  /* set all values to '0' */
  for (x = 0; x < none_zero_num; x++)
  {
    row[x] = 0;
    col[x] = 0;
    val_input[x] = 0.0;
  }
  
  /* put sources in field */ 
 for (x = 0; x < none_zero_num; x++)
  {
   s =  fscanf(f, "%i %i %f\n", &source_x, &source_y, &source_val);
    if (s==3)
    {
      row[x] = source_x;
      col[x] = source_y;
      val_input[x] = source_val;
    //  printf("HeHe\n");
    //  printf("the ith is: %i, %i, %f\n", source_x, source_y, source_val);
    }
  } 
  fclose(f);

/* for(x = 0; x < none_zero_num; x++)
 {
   printf("the %i th is: %i, %i, %f\n", x, row[x], col[x], val_input[x]);
 }*/
}



void CSR_Transformation()
{ 
  int each_row_counter[row_num+1];
  int Matrix_row_counter[row_num+1];
  int counter = 0;
  int i, j;
  int each_val_col;
  int CSR_counter = 0;


 // printf("Captainking"); 

  //set all the row counter = 0;
  for(i = 0; i < row_num +1; i++)
  {
     each_row_counter[i] = 0;
     Matrix_row_counter[i] = 0;
  }


  
   
  for(i = 0; i < none_zero_num; i++)     //count the non_zero value in each row
  {
       j = row[i];
       each_row_counter[j] = each_row_counter[j] + 1;   
  }

  

  if ((val_Matrix = malloc((row_num + 1) * sizeof(val_Matrix))) == NULL)
    Debug("CSR_Transformation : malloc (val_1) failed", 1);
  if ((row_Matrix = malloc((row_num + 1) * sizeof(row_Matrix))) == NULL)
    Debug("CSR_Transformation : malloc (row_1) failed", 1);
  if ((col_Matrix = malloc((row_num + 1) * sizeof(val_Matrix))) == NULL)
    Debug("CSR_Transformation : malloc (val_1) failed", 1);

  for (i = 0; i< row_num + 1; i++)
  {
   if ((row_Matrix[i] = malloc((each_row_counter[i]) * sizeof(*row_Matrix))) == NULL)
     Debug("CSR_Transformation : malloc (row_2) failed", 1);
   if ((col_Matrix[i] = malloc((each_row_counter[i]) * sizeof(*col_Matrix))) == NULL)
     Debug("CSR_Transformation : malloc (col) failed", 1);
   if ((val_Matrix[i] = malloc((each_row_counter[i]) * sizeof(*val_Matrix))) == NULL)
     Debug("CSR_Transformation : malloc (val_2) failed", 1);
  } 
   
  for (i = 1; i < row_num ; i++)
  {
     for (j = 1; j < each_row_counter[i] ; j++)
     {
       val_Matrix[i][j] = 0.0;
       row_Matrix[i][j] = 0;  // none sense, or = i;
       col_Matrix[i][j] = 0;
      }
  }
 
  for (i = 0; i < none_zero_num; i++)
  {
    j = row[i];
    Matrix_row_counter[j]++;
    val_Matrix[j][Matrix_row_counter[j]] = val_input[i];
    row_Matrix[j][Matrix_row_counter[j]] = row[i];
    col_Matrix[j][Matrix_row_counter[j]] = col[i];
  }
  printf("!^_^! ^_^! ^_^! ^_^! ^_^!\n");  
 /* for (i = 1; i < row_num + 1; i++)
  {
     for (j = 1; j < each_row_counter[i] + 1; j++)
     {
       printf("%i, %i: %f\n", row_Matrix[i][j], col_Matrix[i][j], val_Matrix[i][j]);
      }
  } */ 
  
  //allocate memory for the output data
  if ((row_ptr = malloc((row_num + 1) * sizeof(*row_ptr))) == NULL)
    Debug("Read_Matrix : malloc(row) failed", 1);
  if ((col_idx = malloc(none_zero_num * sizeof(*col_idx))) == NULL)
    Debug("Read_Matrix : malloc(col) failed", 1);
  if ((val_output = malloc(none_zero_num * sizeof(*val_output))) == NULL)
    Debug("Read_Matrix : malloc(val_input) failed", 1);
 
  for(i = 0; i< row_num + 1; i++)
    row_ptr[i] = 0;
  //set the row_ptr, col_idx and val_output 0
  for(i = 0; i< none_zero_num; i++)
  {
    col_idx[i] = 0;
    val_output[i] = 0;
  }
  

  for (i = 1; i < row_num +1; i++)
  {
   row_ptr[i] = each_row_counter[i] + row_ptr[i-1];
  } 
  
  for (i = 1; i < row_num + 1; i++)
  {
     for (j = 1; j < each_row_counter[i] + 1; j++)
     { 
       val_output[CSR_counter] = val_Matrix[i][j];
       col_idx[CSR_counter] = col_Matrix[i][j];    
       CSR_counter++;
      }
  }  

  
   

/*  printf("the row_ptr is \n");
  for(i = 0; i < row_num + 1; i++)
  {
    printf("%i\n", row_ptr[i]);
  } 
  printf("the val_output is \n");
  for(i = 0; i < none_zero_num; i++)
  {
    printf("%f\n", val_output[i]);
  } 
  printf("the col_idx is \n");
  for(i = 0; i < none_zero_num; i++)
  {
    printf("%i\n", col_idx[i]);
  } */
}






void Write_Grid()
{
  int x, y;
  FILE *f;

  if ((f = fopen("csr_matrix.dat", "w")) == NULL)
    Debug("Write_Grid : fopen failed", 1);

  Debug("Write_Grid", 0);
  fprintf(f, "csr_matrix_formate(row*column, none_zero_num): %i*%i, %i\n", row_num, col_num, none_zero_num);
  fprintf(f, "row_ptr:\n");
  for (x = 0; x < row_num + 1; x++)
  {  
   fprintf(f, "%i\n", row_ptr[x]);
  }
 
  fprintf(f, "col_idx and val:\n");

  for (x = 0; x < none_zero_num; x++)
     fprintf(f, "%i %f\n", col_idx[x]-1, val_output[x]);
  
  fclose(f);
}

void Clean_Up()
{
  Debug("Clean_Up", 0);

  free(row);
  free(col);
  free(val_input);
  free(row_Matrix);
  free(val_Matrix);
  free(col_Matrix);
}

int main(int argc, char **argv)
{ 
  start_timer();

  Read_Matrix();

  CSR_Transformation();

  Write_Grid();

  print_timer();

  Clean_Up();

  return 0;
}
