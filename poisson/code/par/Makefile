CC=mpicc
SRC=MPI_Poisson.c
EXEC=MPI_Poisson

NODES_NUM=1
CORES_NUM=4
P_GRID_X=4
P_GRID_Y=1
RELAX_FACTOR=1.95 # default value; can be reset from command line

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	prun -v -np $(NODES_NUM) -$(CORES_NUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC) $(P_GRID_X) $(P_GRID_Y) $(RELAX_FACTOR)

clean:
	rm -f output*.dat $(EXEC) profile*.csv error*.csv

profile:
	make RELAX_FACTOR=1.90 run
	make RELAX_FACTOR=1.91 run
	make RELAX_FACTOR=1.92 run
	make RELAX_FACTOR=1.93 run
	make RELAX_FACTOR=1.94 run
	make RELAX_FACTOR=1.95 run
	make RELAX_FACTOR=1.96 run
	make RELAX_FACTOR=1.97 run
	make RELAX_FACTOR=1.98 run
	make RELAX_FACTOR=1.99 run

.PHONY: compile run clean profile
