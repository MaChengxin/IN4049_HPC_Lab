CC=mpicc
SRC=MPI_Poisson.c
EXEC=MPI_Poisson

NODESNUM=1
CORESNUM=2

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	prun -v -np $(NODESNUM) -$(CORESNUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC)

clean:
	rm -f output*.dat $(EXEC)
	
.PHONY: compile run clean
