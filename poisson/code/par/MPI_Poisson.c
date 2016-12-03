/*
 * MPI_Poisson.c
 * 2D Poison equation solver
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEBUG 0
#define PROFILING 1

#define max(a,b) ((a)>(b)?a:b)
#define ceildiv(a,b) (1+((a)-1)/(b))

enum
{
	X_DIR, Y_DIR
};

/* global variables */
int gridsize[2];
double precision_goal;		/* precision_goal of solution */
int max_iter;			    /* maximum number of iterations alowed */

/* MPI-related global variables */
/* process specific variables */
int proc_rank;                  /* rank of current process */
int proc_coord[2];              /* coordinates of current process in process grid */
int proc_top, proc_right, proc_bottom, proc_left; /* ranks of neigboring procs */

int P;                          /* total number of processes */
int P_grid[2];                  /* process grid dimensions */
MPI_Comm grid_comm;             /* grid COMMUNICATOR */
MPI_Status status;
MPI_Datatype border_type[2];

/* benchmark related variables */
clock_t ticks;			/* number of systemticks */
int timer_on = 0;		/* is timer running? */
double wtime;               /*wallclock time*/

/* local grid related variables */
double **phi;			/* grid */
int **source;			/* TRUE if subgrid element is a source */
int dim[2];			/* grid dimensions */
int offset[2];

void Setup_Grid();
double Do_Step(int parity, double omega);
void Solve(int argc, char **argv);
void Write_Grid();
void Clean_Up();
void Setup_Proc_Grid(int argc, char **argv);
void Setup_MPI_Datatypes();
void Exchange_Borders();

void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();

void start_timer()
{
	if (!timer_on)
	{
		/*Each task, when reaching the MPI_Barrier call, blocks until all tasks in the group reach the same MPI_Barrier call.
		Then all tasks are free to proceed.*/
		MPI_Barrier(grid_comm); /*assures all processes start simultaneously with their timing activities*/
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
	FILE *fprof;
	char profile_filename[40];
	sprintf(profile_filename, "profile_%i.csv", gridsize[X_DIR]);
	if ((fprof = fopen(profile_filename, "a")) == NULL)
		Debug("print_timer : fopen failed", 1);

	if (timer_on)
	{
		stop_timer();
		if (!PROFILING)
			printf("Rank of process: %i\t Elapsed Wtime: %14.6f s \t (%5.1f%% CPU)\n",
			       proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
		else
			fprintf(fprof, "Rank of process:\t %i\t Elapsed Wtime (s):\t %.6f\n",
			        proc_rank, wtime);
		resume_timer();
	}
	else
	{
		if (!PROFILING)
			printf("Rank of process: %i\t Elapsed Wtime: %14.6f s \t (%5.1f%% CPU)\n",
			       proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
		else
			fprintf(fprof, "Rank of process:\t %i\t Elapsed Wtime (s):\t %.6f\n",
			        proc_rank, wtime);
	}

	fclose(fprof);
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
	int upper_offset[2];
	double source_x, source_y, source_val;
	FILE *f;

	Debug("Setup_Subgrid", 0);

	if (proc_rank == 0) /* only process 0 may execute this if-statement */
	{
		f = fopen("input.dat", "r");
		if (f == NULL)
			Debug("Error opening input.dat", 1);
		fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
		fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
		fscanf(f, "precision goal: %lf\n", &precision_goal);
		fscanf(f, "max iterations: %i\n", &max_iter);
	}
	MPI_Bcast(&gridsize, 2, MPI_INT, 0, grid_comm); /* broadcast the array gridsize in one call */
	MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm); /* broadcast precision_goal */
	MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm); /* broadcast max_iter */

	/* Calculate top left corner coordinates of local grid */
	offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / P_grid[X_DIR];
	offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / P_grid[Y_DIR];
	upper_offset[X_DIR] = gridsize[X_DIR] * (proc_coord[X_DIR] + 1) / P_grid[X_DIR];
	upper_offset[Y_DIR] = gridsize[Y_DIR] * (proc_coord[Y_DIR] + 1) / P_grid[Y_DIR];

	/* Calculate dimensions of local grid */
	dim[Y_DIR] = upper_offset[Y_DIR] - offset[Y_DIR];
	dim[X_DIR] = upper_offset[X_DIR] - offset[X_DIR];

	/* Add space for rows/columns of neighboring grid */
	dim[Y_DIR] += 2;
	dim[X_DIR] += 2;

	/* allocate memory */
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

	/* set all values to '0' */
	for (x = 0; x < dim[X_DIR]; x++)
		for (y = 0; y < dim[Y_DIR]; y++)
		{
			phi[x][y] = 0.0;
			source[x][y] = 0;
		}

	/* put sources in field */
	do
	{
		if (proc_rank == 0) /* only process 0 may scan next line of input */
			s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);

		MPI_Bcast(&s, 1, MPI_INT, 0, grid_comm); /* The return value of this scan is broadcast even though it is not input data */
		if (s == 3)
		{
			MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, grid_comm); /* broadcast source_x */
			MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, grid_comm); /* broadcast source_y */
			MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, grid_comm); /* broadcast source_val */
			x = gridsize[X_DIR] * source_x;
			y = gridsize[Y_DIR] * source_y;
			x += 1;
			y += 1;
			x = x - offset[X_DIR];
			y = y - offset[Y_DIR];
			if ( x > 0 && x < dim[X_DIR] - 1 &&
			        y > 0 && y < dim[Y_DIR] - 1)
			{	/* indices in domain of this process */
				phi[x][y] = source_val;
				source[x][y] = 1;
			}
		}
	}
	while (s == 3);

	if (proc_rank == 0) /* only process 0 may close the file */
		fclose(f);
}

double Do_Step(int parity, double omega)
{
	int x, y;
	double old_phi;
	double max_err = 0.0;

	/* calculate interior of grid */
	for (x = 1; x < dim[X_DIR] - 1; x++)
		for (y = 1; y < dim[Y_DIR] - 1; y++)
			if ((x + offset[X_DIR] + y + offset[Y_DIR]) % 2 == parity && source[x][y] != 1)
			{
				old_phi = phi[x][y];
				phi[x][y] = (phi[x + 1][y] + phi[x - 1][y] +
				             phi[x][y + 1] + phi[x][y - 1]) * 0.25 * omega
				            +
				            old_phi * (1 - omega);
				if (max_err < fabs(old_phi - phi[x][y]))
					max_err = fabs(old_phi - phi[x][y]);
			}

	return max_err;
}

void Solve(int argc, char **argv)
{
	int count = 0;
	double delta;
	double delta1, delta2;
	double global_delta;
	double omega;
	int border_exchange_factor;
	double base_time, curr_elapsed_time;

	FILE *fprof;
	FILE *f_err;

	char profile_filename[40];
	sprintf(profile_filename, "profile_%i.csv", gridsize[X_DIR]);
	if ((fprof = fopen(profile_filename, "a")) == NULL)
		Debug("Solve : fopen failed", 1);

	char error_filename[40];
	sprintf(error_filename, "error_%i.csv", gridsize[X_DIR]);
	if ((f_err = fopen(error_filename, "a")) == NULL)
		Debug("Solve : fopen failed", 1);

	if (argc > 3)
	{
		omega =  atof(argv[3]);
		if (proc_rank == 0) /* only process 0 may execute this if-statement such that it is only printed once */
		{
			if (!PROFILING)
				printf("The overrelaxation coefficient (omega) passed from command line is %.3f\n", omega);
			else
				fprintf(fprof, "\nThe overrelaxation coefficient (omega) passed from command line is\t %.3f\n", omega);
		}
	}

	if (argc > 4)
	{
		border_exchange_factor =  atoi(argv[4]);
		if (proc_rank == 0) /* only process 0 may execute this if-statement such that it is only printed once */
		{
			if (!PROFILING)
				printf("The border exchange factor passed from command line is %i\n", border_exchange_factor);
			else
				fprintf(fprof, "The border exchange factor passed from command line is\t %i\n", border_exchange_factor);
		}
	}

	Debug("Solve", 0);

	/* give global_delta a higher value then precision_goal */
	global_delta = 2 * precision_goal;

	while (global_delta > precision_goal && count < max_iter)
	{
		int i = 0;

		Debug("Do_Step 0", 0);
		for (i = 0; i < border_exchange_factor; i++)
		{
			delta1 = Do_Step(0, omega);
		}
		Exchange_Borders();

		Debug("Do_Step 1", 0);
		for (i = 0; i < border_exchange_factor; i++)
		{
			delta2 = Do_Step(1, omega);
		}
		Exchange_Borders();

		delta = max(delta1, delta2);
		if (count % 1 == 0) /* switch of adjusting the frequency of calling MPI_Allreduce */
			MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, grid_comm);
		count++;

		if (PROFILING && proc_rank == 0)
		{
			if (count == 1)
			{
				// fprintf(f_err, "The overrelaxation coefficient (omega) passed from command line is\t %.3f\n", omega);
				// fprintf(f_err, "The border exchange factor passed from command line is\t %i\n", border_exchange_factor);
				base_time = MPI_Wtime();
				curr_elapsed_time = 0;
				fprintf(f_err, "Number of iterations:\t %i\t Error:\t %.6f\t Elapsed Wtime:\t %14.6f\n", count, delta, curr_elapsed_time);
			}
			else
			{
				curr_elapsed_time = MPI_Wtime() - base_time;
				fprintf(f_err, "Number of iterations:\t %i\t Error:\t %.6f\t Elapsed Wtime:\t %14.6f\n", count, delta, curr_elapsed_time);
			}
		}
	}

	if (!PROFILING)
		printf("Rank of process: %i\t Number of iterations: %i\n", proc_rank, count);
	else
		fprintf(fprof, "Rank of process:\t %i\t Number of iterations:\t %i\n", proc_rank, count);

	fclose(fprof);
	fclose(f_err);
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

		/* allocate memory for receiving phi */
		max_griddim[X_DIR] = ceildiv(gridsize[X_DIR], P_grid[X_DIR]) + 2;
		max_griddim[Y_DIR] = ceildiv(gridsize[Y_DIR], P_grid[Y_DIR]) + 2;

		if ((sub_phi = malloc(max_griddim[X_DIR] * sizeof(*sub_phi))) == NULL)
			Debug("Write_Grid : malloc(sub_phi) failed", 1);
		if ((sub_phi[0] = malloc(max_griddim[X_DIR] * max_griddim[Y_DIR] *
		                         sizeof(**sub_phi))) == NULL)
			Debug("Write_Grid : malloc(sub_phi) failed", 1);

		/* write data for process 0 to disk */
		for (x = 1; x < dim[X_DIR] - 1; x++)
			for (y = 1; y < dim[Y_DIR] - 1; y++)
				fprintf(f, "%i %i %f\n", offset[X_DIR] + x, offset[Y_DIR] + y, phi[x][y]);

		/* receive and write data form other processes */
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
					fprintf(f, "%i %i %f\n", grid_offs[X_DIR] + x, grid_offs[Y_DIR] + y, sub_phi[x][y]);
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
}

void Clean_Up()
{
	Debug("Clean_Up", 0);

	free(phi[0]);
	free(phi);
	free(source[0]);
	free(source);
}

void Setup_Proc_Grid(int argc, char **argv)
{
	int wrap_around[2];
	int reorder;

	Debug("My_MPI_Init", 0);

	/* Retrieve the number of processes*/
	MPI_Comm_size(MPI_COMM_WORLD, &P); /*Find out how many processes there are*/

	/* Calculate the number of processes per column and per row for the grid */
	if (argc > 2)
	{
		P_grid[X_DIR] = atoi(argv[1]);
		P_grid[Y_DIR] = atoi(argv[2]);
		if (P_grid[X_DIR] * P_grid[Y_DIR] != P)
			Debug("ERROR : Process grid dimensions do not match with P", 1);
	}
	else
		Debug("ERROR : Wrong parameter input", 1);

	/* Create process topology (2D grid) */
	wrap_around[X_DIR] = 0;
	wrap_around[Y_DIR] = 0;       /* Do not connect first and last process */
	reorder = 1;                  /* Reorder process ranks */
	MPI_Cart_create(MPI_COMM_WORLD, 2, P_grid, wrap_around, reorder, &grid_comm); /*Creates a new communicator: grid_comm*/

	/* Retrieve new rank and carthesian coordinates of this process */
	MPI_Comm_rank(grid_comm, &proc_rank); /* Rank of process in new communicator */
	MPI_Cart_coords(grid_comm, proc_rank, 2, proc_coord); /* Coordinates of a process in the new communicator */

	if (!PROFILING)
		printf("The coordinates of process %i is (%i,%i)\n", proc_rank, proc_coord[X_DIR], proc_coord[Y_DIR]);

	/* Calculate ranks of neighbouring processes */
	MPI_Cart_shift(grid_comm, Y_DIR, 1, &proc_top, &proc_bottom); /* Rank of processes proc_top and proc_bottom */
	MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right); /* Rank of processes proc_left and proc_right */

	if (DEBUG)
		printf("(%i) top %i, right %i, bottom %i, left %i\n",
		       proc_rank, proc_top, proc_right, proc_bottom, proc_left);
}

void Setup_MPI_Datatypes()
{
	Debug("Setup_MPI_Datatypes", 0);

	/* Datatype for vertical data exchange (Y_DIR) */
	MPI_Type_vector(dim[X_DIR] - 2, 1, dim[Y_DIR], MPI_DOUBLE, &border_type[Y_DIR]);
	MPI_Type_commit(&border_type[Y_DIR]);

	/* Datatype for horizontal data exchange (X_DIR) */
	MPI_Type_vector(dim[Y_DIR] - 2, 1, 1, MPI_DOUBLE, &border_type[X_DIR]);
	MPI_Type_commit(&border_type[X_DIR]);
}

void Exchange_Borders()
{
	Debug("Exchange_Borders", 0);

	MPI_Sendrecv( &phi[1][1], 1, border_type[Y_DIR], proc_top, 0,
	              &phi[1][dim[Y_DIR] - 1], 1, border_type[Y_DIR], proc_bottom, 0,
	              grid_comm, &status); /* all traffic in direction "up" */

	MPI_Sendrecv( &phi[1][dim[Y_DIR] - 2], 1, border_type[Y_DIR], proc_bottom, 0,
	              &phi[1][0], 1, border_type[Y_DIR], proc_top, 0,
	              grid_comm, &status); /* all traffic in direction "down" */

	MPI_Sendrecv( &phi[1][1], 1, border_type[X_DIR], proc_left, 0,
	              &phi[dim[X_DIR] - 1][1], 1, border_type[X_DIR], proc_right, 0,
	              grid_comm, &status); /* all traffic in direction "left" */

	MPI_Sendrecv( &phi[dim[X_DIR] - 2][1], 1, border_type[X_DIR], proc_right, 0,
	              &phi[0][1], 1, border_type[X_DIR], proc_left, 0,
	              grid_comm, &status); /* all traffic in direction "right" */
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	Setup_Proc_Grid(argc, argv);

	start_timer();

	Setup_Grid();

	Setup_MPI_Datatypes();

	Solve(argc, argv);

	Write_Grid();

	print_timer();

	Clean_Up();

	MPI_Finalize();

	return 0;
}
