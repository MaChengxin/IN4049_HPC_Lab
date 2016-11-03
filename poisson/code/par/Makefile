CC=mpicc
SRC=MPI_Poisson.c
EXEC=MPI_Poisson

NODESNUM=1
CORESNUM=4

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	prun -v -np $(NODESNUM) -$(CORESNUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC) 2 2

clean:
	rm -f output*.dat $(EXEC)
	
.PHONY: compile run clean
