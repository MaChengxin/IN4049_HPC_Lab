CC=mpicc
SRC=MPI_Poisson.c
EXEC=MPI_Poisson

NODES_NUM=1
CORES_NUM=4
P_GRID_X=4
P_GRID_Y=1

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	prun -v -np $(NODES_NUM) -$(CORES_NUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC) $(P_GRID_X) $(P_GRID_Y)

clean:
	rm -f output*.dat $(EXEC)
	
.PHONY: compile run clean
