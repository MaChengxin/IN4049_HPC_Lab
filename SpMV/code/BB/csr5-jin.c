/*CSR5-Formation  
 *   By Jianbing Jin   2016-Jan-6
 *  */

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

void CSR5_Transformation();
void tile_ptr_computing();
void tile_val_computing();
void tile_dexcription();

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
//float *val;


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


void CSR5_Transformation()
{
   tile_ptr_computing();
   tile_val_computing();
   tile_dexcription();
}

void tile_dexcription()
{
   int i, j, m;
   int count;
  //allocate memory for the bit_flag[][]
   if ((tile_bit_flag = malloc((p - 1) * sizeof(*tile_bit_flag))) == NULL)    // setup p-1 tiles
     Debug("CSR5_Transformation_tile_bit_flag : malloc (tile_bit_flag) failed", 1);
   for (i = 0; i< (p - 1); i++)
   {
    if ((tile_bit_flag[i] = malloc((Sigma) * sizeof(**tile_bit_flag))) == NULL)  // every tile has Sigma row
     Debug("CSR5_Transformation_tile_bit_flag: malloc (tile_bit_flag) failed", 1);
    for (j = 0; j < Sigma; j++)
    {
     if ((tile_bit_flag[i][j] = malloc((w) * sizeof(***tile_bit_flag))) == NULL) // every row in the  tile has w column
       Debug("CSR5_Transformation_tile_bit_flag: malloc (tile_bit_flag) failed", 1);
    }
   }



   for (count = 0; (count< (p-1) * Sigma * w); count++)
   { 
    // computer the corresponding index i j m for each count;
    i = count / (Sigma * w);  // tile index
    j = (count % (Sigma * w)) / Sigma; //
    m = (count % (Sigma * w)) % Sigma; //
    tile_bit_flag[i][m][j] = False;  //as the tile 
   }

  for(count = 0; count< row_num; count++)
  {
    i = row_ptr[count] / (Sigma * w);
    j = (row_ptr[count] % (Sigma * w)) / Sigma;
    m = (row_ptr[count] % (Sigma * w)) % Sigma; 

    tile_bit_flag[i][m][j] = True;
  } 

  for (i = 0; i< (p - 1); i++)
  {
    tile_bit_flag[i][0][0] = True;
  } 

  for (i = 0; i< (p - 1); i++)
  {
    for (j = 0; j < Sigma; j++)
    {
      for(m = 0; m < w; m++)
      {
        printf("tile_bit_flag[%i][%i][%i] is %i\n", i, m, j,  tile_bit_flag[i][m][j]);
      }
    }
  } 
   ///allocate memory for the y_offset and seg_offset
  if ((tile_y_offset = malloc((p - 1) * sizeof(*tile_y_offset))) == NULL)  
    Debug("CSR5_Transformation_tile_y_offset : malloc (tile_y_offset) failed", 1);
  if ((tile_seg_offset = malloc((p - 1) * sizeof(*tile_y_offset))) == NULL)  
    Debug("CSR5_Transformation_tile_seg_offset : malloc (tile_seg_offset) failed", 1);
  for (i = 0; i< p - 1; i++)
  {
    if ((tile_y_offset[i] = malloc((w) * sizeof(**tile_y_offset))) == NULL)
      Debug("CSR5_Transformation_tile_y_offset: malloc (tile_y_offset) failed", 1);
    if ((tile_seg_offset[i] = malloc((w) * sizeof(*tile_seg_offset))) == NULL)  
      Debug("CSR5_Transformation_tile_seg_offset : malloc (tile_seg_offset) failed", 1);
  } 
  //compute the y_offset//
  for (i = 0; i< (p - 1); i++)
  { 
    int bais  = 0;
    for (m = 0; m < w; m++)
    {     
      tile_y_offset[i][m] = bais;
      for (j = 0; j < Sigma; j++)
      { 
        bais = bais + tile_bit_flag[i][j][m];
      }
      printf("tile_y_offset[%i][%i] is %i\n", i, m, tile_y_offset[i][m]); 
    }

   //compute the seg_offset//
   for (m = 0; m < w; m++)
   {  
     int seg = 0;    
     for (j = 0; j < Sigma; j++)
     {
       seg = max(tile_bit_flag[i][j][m], seg);
     }
     tile_seg_offset[i][m] = 1 - seg;
   }
   
   for (m = 0; m < w -1; m++)
   {
     tile_seg_offset[i][m] = tile_seg_offset[i][m+1]; 
   }
   tile_seg_offset[i][m] = 0;
   for (m = 0; m < w; m++)
   {
     printf("tile_seg_offset[%i][%i] is %i\n", i, m, tile_seg_offset[i][m]); 
   } 
  } 
   
  //allocate memory for the empty_offset
  if ((tile_empty_offset = malloc((p - 1) * sizeof(*tile_empty_offset))) == NULL)  
    Debug("CSR5_Transformation_tile_empty_offset : malloc (tile_empty_offset) failed", 1);  
  
  int empty_tile_true_counter[p - 1]; //calculate the number of "True" in each tile with empty row!!!
  for (i = 0; i< (p - 1); i++)
  { 
    empty_tile_true_counter[i] = 0;
    if(tile_ptr[i] <= 0)
      for (m = 0; m < w; m++)
      {
        for(j = 0; j < Sigma; j++)
        {
          if(tile_bit_flag[i][j][m] == True)
            empty_tile_true_counter[i]++; 
        }
      }
     printf("empty_tile_true_couter[%i] is %i\n", i, empty_tile_true_counter[i]);
  }
   
   for (i = 0; i< (p - 1); i++)
   {                            //empty_tile_true_counter[i]
    if ((tile_empty_offset[i] = malloc(empty_tile_true_counter[i] * sizeof(**tile_empty_offset))) == NULL)
      Debug("CSR5_Transformation_tile_empty_offset: malloc (tile_empty_offset) failed", 1);  
    empty_tile_true_counter[i] = 0; // it will be used in the next program
   }
  
  for (i = 0; i< (p - 1); i++)
  { 
    int row_ptr_counter = 0;
    int tid;
    if(tile_ptr[i] <= 0)
      for (m = 0; m < w; m++)
      {
        for(j = 0; j < Sigma; j++)
        {
          if(tile_bit_flag[i][j][m] == True)
          {
             
            tid = i * Sigma * w + m * w + j;
            for(row_ptr_counter = 0; row_ptr_counter < row_num; row_ptr_counter++)
            {
              if ((tid >= row_ptr[row_ptr_counter]) && (tid <= row_ptr[row_ptr_counter+1]))
              {   
                printf("hehehehehehhehehhe\n");
                tile_empty_offset[i][empty_tile_true_counter[i]] = row_ptr_counter + tile_ptr[i]; // equal to tile_empty[][] = row_ptr_counter - remove_sign(tile_ptr[i])
               // printf("tile_empty_offset[%i][%i] is %i\n", i, empty_tile_true_counter[i], tile_empty_offset[i][empty_tile_true_counter[i]]);
              }
            }
            empty_tile_true_counter[i]++;
          }
        }
      }
  }

  for (i = 0; i< (p - 1); i++)
  {
    if(tile_ptr[i] <= 0)
    {
     for(j = 0; j < empty_tile_true_counter[i]; j++)
     {
       printf("tile_empty_offset[%i][%i] is %i\n", i, j, tile_empty_offset[i][j]);
     }
    }
  }



}

void tile_val_computing()
{
  int i, j, m;
  int count;
 
  if ((tile_val = malloc((p-1) * sizeof(*tile_val))) == NULL)
    Debug("CSR5_Transformation_tile_val : malloc (tile_val) failed", 1);
  if ((tile_col_idx = malloc((p-1) * sizeof(*tile_col_idx))) == NULL)
    Debug("CSR5_Transformation_tile_col_idx : malloc (tile_col_idx) failed", 1);

  for (i = 0; i< p-1; i++)
  {
    if ((tile_val[i] = malloc((Sigma) * sizeof(**tile_val))) == NULL)
     Debug("CSR5_Transformation_tile_val: malloc (tile_val[i]) failed", 1);
    if ((tile_col_idx[i] = malloc((Sigma) * sizeof(**tile_col_idx))) == NULL)
     Debug("CSR5_Transformation_tile_col_idx: malloc (tile_col_idx[i]) failed", 1);
    
    for (j = 0; j < Sigma; j++)
    {
     if ((tile_val[i][j] = malloc((w) * sizeof(***tile_val))) == NULL)
       Debug("CSR5_Transformation_tile_val : malloc (tile_val[i][j]) failed", 1);
     if ((tile_col_idx[i][j] = malloc((w) * sizeof(***tile_col_idx))) == NULL)
       Debug("CSR5_Transformation_tile_col_idx : malloc (tile_col_idx[i][j]) failed", 1);
    }
  } 

 for (count = 0; (count< (p-1) * Sigma * w); count++)
 { 
   // computer the corresponding index i j m for each count;
   i = count / (Sigma * w);
   j = (count % (Sigma * w)) / Sigma;  //the column index 
   m = (count % (Sigma * w)) % Sigma;  //the row index

   tile_val[i][m][j] = val_output[count];
   tile_col_idx[i][m][j] = col_idx[count];
 }

 
 for (i = 0; i< p-1; i++)
 {
    for(j = 0; j < Sigma; j++)
    {
      for (m = 0; m < w; m++)
      {
   //     printf("tile_val[%i][%i][%i] is %f\n", i, m, j,  tile_val[i][m][j]);
    //    printf("tile_col_idx[%i][%i][%i] is %i\n", i, m, j,  tile_col_idx[i][m][j]);
      }
    }
 } 

}



void tile_ptr_computing()
{
  //int p;
  int bnd;
  int tid, i;

  if ((none_zero_num % (w * Sigma)) > 0)
    p = (none_zero_num / (w * Sigma)) + 1;
  else
    p = none_zero_num / (w * Sigma);

  if ((tile_ptr = (int *)malloc((p + 1) * sizeof(*tile_ptr))) == NULL)
    Debug("Tile_ptr_computing : malloc(tile_ptr) failed", 1);
  
  tile_ptr[p] = row_num;
  for(tid = 0; tid <= p; tid++)
  {
    bnd  = tid * w * Sigma;
    for (i = 0; i <= row_num; i++)
    { 
      if (bnd < row_ptr[i+1] && bnd >= row_ptr[i])
        tile_ptr[tid] = i; 
    }     
   // printf("tile_ptr %i is %i\n", tid,  tile_ptr[tid]);
  } 
  for(tid = 0; tid < p; tid++)  //check whether there is a empty row in the tile[i]
  {
    for(i = tile_ptr[tid]; i <= tile_ptr[tid+1]; i++)
    {
      if(row_ptr[i] == row_ptr[i+1])
      {
        tile_ptr[tid] = -tile_ptr[tid];
        break;
      }
    }
  }
  for(tid = 0; tid <= p; tid++)
  { 
  //  printf("tile_ptr %i is %i\n", tid,  tile_ptr[tid]);
  } 
   
}
void Sigma_computing()
{
  int r = 4;
  int s = 32;
  int t = 256;
  int u = 4;
  int division_result;

  division_result = (int)(none_zero_num/row_num);
  if (division_result < r)
     Sigma = r;
  else 
     if (division_result <= s)
       Sigma = division_result;
     else
       if (division_result <= t)
         Sigma = s;
       else
         Sigma = u;
  
  printf("Sigma is %i\n", division_result);
}


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

  
  if ((row_Matrix = malloc((row_num + 1) * sizeof(*row_Matrix))) == NULL)
    Debug("CSR_Transformation : malloc (row_1) failed", 1);
  if ((col_Matrix = malloc((row_num + 1) * sizeof(*col_Matrix))) == NULL)
    Debug("CSR_Transformation : malloc (col_1) failed", 1);
  if ((val_Matrix = malloc((row_num + 1) * sizeof(*val_Matrix))) == NULL)
    Debug("CSR_Transformation : malloc (val_1) failed", 1);
  for (i = 0; i< row_num + 1; i++)
  {
   if ((row_Matrix[i] = malloc((each_row_counter[i]) * sizeof(**row_Matrix))) == NULL)
     Debug("CSR_Transformation : malloc (row_2) failed", 1);
   if ((col_Matrix[i] = malloc((each_row_counter[i]) * sizeof(**col_Matrix))) == NULL)
     Debug("CSR_Transformation : malloc (col) failed", 1);
   if ((val_Matrix[i] = malloc((each_row_counter[i]) * sizeof(**val_Matrix))) == NULL)
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
       col_idx[CSR_counter] = col_Matrix[i][j] - 1;  // the fortran is different from c language for the strage of matrix /////////////////////////////////    
       CSR_counter++;
      }
  }  
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
     fprintf(f, "%i %f\n", col_idx[x], val_output[x]);
  
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
  free(val_output);
  free(col_idx);
  free(row_ptr);
}

int main(int argc, char **argv)
{ 
  start_timer();

  Read_Matrix();

  CSR_Transformation();
  
  Sigma_computing();
  
  CSR5_Transformation();

  Write_Grid();

  print_timer();

  Clean_Up();

  return 0;
}

