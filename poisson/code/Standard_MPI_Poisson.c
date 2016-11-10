/*
 * SEQ_Poisson.c
 * 2D Poison equation solver
 */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEBUG 0

#define max(a,b) ((a)>(b)?a:b)
#define ceildiv(a,b) (1+((a)-1)/(b))

enum
{
  X_DIR, Y_DIR
};

/*process specific variables */
int proc_rank;
int proc_coord[2];                                 // show the coordinates of the current process
int proc_top, proc_right, proc_bottom, proc_left;  // ranks of the neigboring procs  

/* global variables */
int gridsize[2];
double precision_goal;		   /* precision_goal of solution */
int max_iter;			   /* maximum number of iterations alowed */
int offset[2];                     // for ghost point  
double wtime;

double W_relaxation = 1.95; // the relaxation parameter

int P;                         // total number of the processes
int P_grid[2];                 // process dimensions
MPI_Comm grid_comm;            // grid COMMUNICATOR
MPI_Status status;            
MPI_Datatype border_type[2];         //question for how the exchanging works//
//double global_delta;



/* benchmark related variables */
clock_t ticks;			/* number of systemticks */
int timer_on = 0;		/* is timer running? */
double MPI_Reduce_Time;
/* local grid related variables */    // the grid for the data 
double **phi;			/* grid */
int **source;			/* TRUE if subgrid element is a source */
int dim[2];			/* grid dimensions */

void Setup_MPI_Datatypes();
void Exchange_Borders();    
void Setup_Grid();
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
    MPI_Barrier(grid_comm);
    ticks = clock();
    wtime = MPI_Wtime();
    timer_on = 1;
  }
}

void resume_timer()
{
  if (!timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 1;
  }
}

void stop_timer()
{
  if (timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 0;
  }
}

void print_timer()
{
  if (timer_on)
  {
    stop_timer();
    printf("(%i) Elapsed Wtime: %14.6f s (%5.1f%% CPU)\n",
           proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
    resume_timer();
  }
  else
    printf("(%i) Elapsed Wtime: %14.6f s (%5.1f%% CPU)\n",
           proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);

}

void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("%s\n", mesg);
  if (terminate)
    exit(1);
}

void Setup_Grid()
{
  int x, y, s;
  double source_x, source_y, source_val;
  FILE *f;
  int upper_offset[2];   // used for the ghost point

  Debug("Setup_Subgrid", 0);
  if( proc_rank == 0 )
  {
    f = fopen("input.dat", "r");
    if (f == NULL)
      Debug("Error opening input.dat", 1);
    fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
    fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
    fscanf(f, "precision goal: %lf\n", &precision_goal);
    fscanf(f, "max iterations: %i\n", &max_iter);
  }

  MPI_Bcast(gridsize, 2, MPI_INT, 0, grid_comm);
  MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm);
  MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm);
 
  
  
 ////// Calculate dimensions of local subgrid 
 // dim[X_DIR] = gridsize[X_DIR] + 2;    //use for the SEQ
 // dim[Y_DIR] = gridsize[Y_DIR] + 2;    // and shield on Nov 3
 


 // Calculate top left corner coordinates of local grid 
 // offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / P_grid[X_DIR];
 // offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / P_grid[Y_DIR];
 // upper_offset[X_DIR] = gridsize[X_DIR] * (proc_coord[X_DIR] + 1) / P_grid[X_DIR];
 // upper_offset[Y_DIR] = gridsize[Y_DIR] * (proc_coord[Y_DIR] + 1) / P_grid[Y_DIR];
  offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / P_grid[X_DIR];
  offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / P_grid[Y_DIR];
  upper_offset[X_DIR] = gridsize[X_DIR] * (proc_coord[X_DIR] + 1) / P_grid[X_DIR];
  upper_offset[Y_DIR] = gridsize[Y_DIR] * (proc_coord[Y_DIR] + 1) / P_grid[Y_DIR];
  
  // Calculate dimensions of local grid 
  dim[Y_DIR] = upper_offset[Y_DIR]-offset[Y_DIR];
  dim[X_DIR] = upper_offset[X_DIR]-offset[X_DIR];
  // Add space for rows of neighbour 
  dim[Y_DIR] += 2;
  dim[X_DIR] += 2;
  
  

  // allocate memory 
  if ((phi = malloc(dim[X_DIR] * sizeof(*phi))) == NULL)
    Debug("Setup_Subgrid : malloc(phi) failed", 1);
  if ((source = malloc(dim[X_DIR] * sizeof(*source))) == NULL)
    Debug("Setup_Subgrid : malloc(source) failed", 1);
  if ((phi[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**phi))) == NULL)
    Debug("Setup_Subgrid : malloc(*phi) failed", 1);
  if ((source[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**source))) == NULL)
    Debug("Setup_Subgrid : malloc(*source) failed", 1);
  for (x = 1; x < dim[X_DIR]; x++)
  {
    phi[x] = phi[0] + x * dim[Y_DIR];
    source[x] = source[0] + x * dim[Y_DIR];
  }
       // shield on 3 Nov

 
  // set all values to '0'
   for (x = 0; x < dim[X_DIR]; x++)
    for (y = 0; y < dim[Y_DIR]; y++)
    {
      phi[x][y] = 0.0;
      source[x][y] = 0;
    }

  //put sources in field 
  do
  { 
    if ( proc_rank == 0 ) // only process 0 may scan next line of input 
      s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);

    MPI_Bcast(&s, 1, MPI_INT, 0, grid_comm);

   // s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);
    if (s==3)
    {
      MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, grid_comm); // broadcast source_x 
      MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, grid_comm); // broadcast source_y 
      MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, grid_comm); // broadcast source_val 
       x = source_x * gridsize[X_DIR];
      y = source_y * gridsize[Y_DIR];
      x += 1;
      y += 1;
     // phi[x][y] = source_val;
     // source[x][y] = 1;
      x = x - offset[X_DIR];   //locate the origin of the point X
      y = y - offset[Y_DIR];   //locate the origin of the point Y
      if ( x>0 && x< dim[X_DIR] -1 && y>0 && y< dim[Y_DIR] -1)
      {      
        phi[x][y] = source_val; // set the never changed value
        source[x][y] = 1;         // se the never changed place 
      }
    }
  }
  while (s==3);

  if ( proc_rank == 0 ) fclose(f);
}

double Do_Step(int parity)
{
  int x, y;
  double old_phi;
  double max_err = 0.0;

  /* calculate interior of grid */
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
     // if ((x + y) % 2 == parity && source[x][y] != 1)
      if ((x + offset[X_DIR] + y + offset[Y_DIR]) % 2 == parity && source[x][y] != 1)
      {
	old_phi = phi[x][y];
	phi[x][y] = (phi[x + 1][y] + phi[x - 1][y] +
		     phi[x][y + 1] + phi[x][y - 1]) * 0.25;
        phi[x][y] = (1-W_relaxation)*old_phi + W_relaxation*phi[x][y];
	if (max_err < fabs(old_phi - phi[x][y]))
	  max_err = fabs(old_phi - phi[x][y]);
      }

  return max_err;
}

void Solve()
{
  int count = 0;
  double delta, global_delta;
  double delta1, delta2;
  int count_10 = 0;
  double MPI_Reduce_EachTime;
  int border_change_threhold = 1, border_change_count = 0;
 // double MPI_Reduce_Time;
  Debug("Solve", 0);

  /* give global_delta a higher value then precision_goal */
   global_delta = 2 * precision_goal;  //  set the initial value
  

  while (global_delta > precision_goal && count < max_iter)
  {
  //  border_change_count++;

    Debug("Do_Step 0", 0);
   
    delta1 = Do_Step(0);
    
   // if (border_change_count == border_change_threhold)
   // { 
     //  MPI_Reduce_EachTime = MPI_Wtime();

       Exchange_Borders();
       
      // MPI_Reduce_EachTime = MPI_Wtime()-MPI_Reduce_EachTime;
       
      // printf("The MPI_Reduce_time is %f\n", MPI_Reduce_EachTime);  
   //    MPI_Reduce_Time += MPI_Reduce_EachTime;   
   // }

  
    //border_change_count++;
  
    Debug("Do_Step 1", 0);
    
    delta2 = Do_Step(1);
   
   // if (border_change_count == 2 * border_change_threhold)
   // {   
       Exchange_Borders();
     //  border_change_count = 0;
   // }

  //  Exchange_Borders();

    
    delta = max(delta1, delta2);
    count++;
  
   // if (count_10 > 9)
   // {
   // MPI_Reduce_EachTime = MPI_Wtime();   
    MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, grid_comm);   // give the max delta to global_delta
   // MPI_Reduce_EachTime = MPI_Wtime()-MPI_Reduce_EachTime;
   // MPI_Reduce_Time += MPI_Reduce_EachTime;    
     // printf("The MPI_Reduce_time is %f\n", MPI_Reduce_Time);
   // }
   // count_10++;
  }
   // printf("The MPI_Reduce_time is %f\n", MPI_Reduce_Time); 
    printf("Number of iterations : %i\n", count);
}

 
void Write_Grid()
{
  int x, y, p;
  int grid_offs[2], grid_dim[2];
  int max_griddim[2];
  double **sub_phi;
  FILE *f;

  Debug("Write_Grid", 0);

  if (proc_rank == 0)
  {
    if ((f = fopen("output.dat", "w")) == NULL)
      Debug("Write_Grid : fopen failed", 1);

   //  allocate memory for receiving phi 
    max_griddim[X_DIR] = ceildiv(gridsize[X_DIR], P_grid[X_DIR]) + 2;
    max_griddim[Y_DIR] = ceildiv(gridsize[Y_DIR], P_grid[Y_DIR]) + 2;

    if ((sub_phi = malloc(max_griddim[X_DIR] * sizeof(*sub_phi))) == NULL)
      Debug("Write_Grid : malloc(sub_phi) failed", 1);
    if ((sub_phi[0] = malloc(max_griddim[X_DIR] * max_griddim[Y_DIR] *
                             sizeof(**sub_phi))) == NULL)
      Debug("Write_Grid : malloc(sub_phi) failed", 1);

    // write data for process 0 to disk 
    for (x = 1; x < dim[X_DIR] - 1; x++)
      for (y = 1; y < dim[Y_DIR] - 1; y++)
        fprintf(f, "%i %i %f\n", offset[X_DIR]+x, offset[Y_DIR]+y, phi[x][y]);

    // receive and write data form other processes 
    for (p = 1; p < P; p++)
    {
      MPI_Recv(grid_offs, 2, MPI_INT, p, 0, grid_comm, &status);
      MPI_Recv(grid_dim, 2, MPI_INT, p, 0, grid_comm, &status);
      MPI_Recv(sub_phi[0], grid_dim[X_DIR] * grid_dim[Y_DIR],
               MPI_DOUBLE, p, 0, grid_comm, &status);

      for (x = 1; x < grid_dim[X_DIR]; x++)
        sub_phi[x] = sub_phi[0] + x * grid_dim[Y_DIR];

      for (x = 1; x < grid_dim[X_DIR] - 1; x++)
        for (y = 1; y < grid_dim[Y_DIR] - 1; y++)
          fprintf(f, "%i %i %f\n", grid_offs[X_DIR]+x, grid_offs[Y_DIR]+y, sub_phi[x][y]);
    }
    free(sub_phi[0]);
    free(sub_phi);

    fclose(f);
  }
  else
  {
    MPI_Send(offset, 2, MPI_INT, 0, 0, grid_comm);
    MPI_Send(dim, 2, MPI_INT, 0, 0, grid_comm);
    MPI_Send(phi[0], dim[Y_DIR] * dim[X_DIR], MPI_DOUBLE, 0, 0, grid_comm);
  }
 /* int x, y, _x, _y;
  FILE *f;
  char filename[40];

  sprintf(filename, "output%i.dat", proc_rank);
  if ((f = fopen(filename, "w")) == NULL)
    Debug("Write_Grid : fopen failed", 1);

  Debug("Write_Grid", 0);

  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
    {
      _x = offset[X_DIR] + x;
      _y = offset[Y_DIR] + y;
      fprintf(f, "%i %i %f\n", _x, _y, phi[x][y]);
   */ 
} 



void Clean_Up()
{
  Debug("Clean_Up", 0);

  free(phi[0]);
  free(phi);
  free(source[0]);
  free(source);
}

void Setup_MPI_Datatypes()
{
  Debug("Setup_MPI_Datatypes",0);

  /*Datatype for vertical data exchange (Y_DIR) */
  MPI_Type_vector(dim[X_DIR] - 2, 1, dim[Y_DIR], MPI_DOUBLE, &border_type[Y_DIR]);
  MPI_Type_commit(&border_type[Y_DIR]); // seems just to commit the Border_type 

  /* datatype for horizonal data exchange (X_DIR)*/
 // MPI_Type_vector(dim[Y_DIR] - 2, 1, dim[X_DIR], MPI_DOUBLE, &border_type[X_DIR]);
  MPI_Type_vector(dim[Y_DIR] - 2, 1, 1, MPI_DOUBLE, &border_type[X_DIR]);
  MPI_Type_commit(&border_type[X_DIR]);
} 

void Exchange_Borders()
{
  //may be i need to change the place with "proc_top" and "proc bottom" in each Sendrecv
  Debug("Exchange_Borders",0);

  MPI_Sendrecv(&phi[1][1], 1, border_type[Y_DIR], proc_top, 0,
               &phi[1][dim[Y_DIR]-1], 1, border_type[Y_DIR], proc_bottom, 0,
               grid_comm, &status); /* all traffic in direction "top" */

  MPI_Sendrecv(&phi[1][dim[Y_DIR]-2], 1, border_type[Y_DIR], proc_bottom, 0,
               &phi[1][0], 1, border_type[Y_DIR], proc_top, 0,
               grid_comm, &status); /* all traffic in direction "bottom" */

  MPI_Sendrecv(&phi[1][1], 1, border_type[X_DIR], proc_left, 0,
               &phi[dim[X_DIR]-1][1], 1, border_type[X_DIR], proc_right, 0,
               grid_comm, &status); /* all traffic in direction "left" */

  MPI_Sendrecv(&phi[dim[X_DIR]-2][1], 1, border_type[X_DIR], proc_right, 0,
               &phi[0][1], 1, border_type[X_DIR], proc_left, 0,
               grid_comm, &status); /* all traffic in direction "right" */
}

void Setup_Proc_Grid(int argc, char **argv)
{ 
  int wrap_around[2];
  int reorder;
//  printf("the argc is %i\n", argc);
  Debug("My_MPI_Init", 0);
  
  /* Retrieve the number of processes P */
  MPI_Comm_size(MPI_COMM_WORLD, &P);    // get the whole processes number
  
  /* Calculate the number of processes per column and per row for the grid */
  if (argc > 2)                        // if the argument counter > 2
  { 
    P_grid[X_DIR] = atoi(argv[1]);  //in the stdlib.h, into the int type
    P_grid[Y_DIR] = atoi(argv[2]);  
    if (P_grid[X_DIR] * P_grid[Y_DIR] != P)
      Debug("ERROR : Proces grid dimensions do not match with P", 1);
  }
  else
    Debug("ERROR : Wrong parameterinput", 1);
  
  /* Create process topology (2D grid) */
  wrap_around[X_DIR] = 0;       
  wrap_around[Y_DIR] = 0;       /* do not connect first and last process */
  reorder = 1;                  /* reorder process ranks */
  //MPI_Cart_create(....., ....., ....., ....., ....., &grid_comm);
  MPI_Cart_create(MPI_COMM_WORLD, 2, P_grid, wrap_around, reorder, &grid_comm);
  
  /* Retrieve new rank and carthesian coordinates of this process */
  MPI_Comm_rank(grid_comm, &proc_rank);
  MPI_Cart_coords(grid_comm, proc_rank, 2, proc_coord);
  
  printf("(%i) hehe:(x,y)=(%i,%i)\n", proc_rank, proc_coord[X_DIR], proc_coord[Y_DIR]);
  
  /* calculate ranks of neighbouring processes */
  MPI_Cart_shift(grid_comm, Y_DIR, 1, &proc_top, &proc_bottom);  ///// rank of processes proc_top and proc_bottom
  MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right);      ///// rank of processes proc_left and proc_right  
  if (DEBUG)  
 // if (!DEBUG)
    printf("(%i) top %i, right %i, bottom %i, left %i\n",
           proc_rank, proc_top, proc_right, proc_bottom, proc_left);
}


int main(int argc, char **argv)
{ 
 
  MPI_Init(&argc, &argv);
  
  Setup_Proc_Grid(argc, argv);
   
  start_timer();
 
  Setup_Grid();
  
  Setup_MPI_Datatypes();

  Solve();

  Write_Grid();

  print_timer();

  Clean_Up();

  MPI_Finalize();

  return 0;
}
