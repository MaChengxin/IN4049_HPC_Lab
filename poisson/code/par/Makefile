CC=mpicc
SRC=MPI_Poisson.c
EXEC=MPI_Poisson

NODES_NUM=1
CORES_NUM=4
P_GRID_X=2
P_GRID_Y=2
RELAX_FACTOR=1.95 # default value; can be reset from command line
BORDER_EXCHANGE_FACTOR=1 # default value; can be reset from command line

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	prun -v -np $(NODES_NUM) -$(CORES_NUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC) $(P_GRID_X) $(P_GRID_Y) $(RELAX_FACTOR) $(BORDER_EXCHANGE_FACTOR)

clean:
	rm -f output*.dat $(EXEC) profile*.csv error*.csv border_exchange_info*.dat

exp-2-2:
	make run RELAX_FACTOR=1.90
	make run RELAX_FACTOR=1.91
	make run RELAX_FACTOR=1.92
	make run RELAX_FACTOR=1.93
	make run RELAX_FACTOR=1.94
	make run RELAX_FACTOR=1.95
	make run RELAX_FACTOR=1.96
	make run RELAX_FACTOR=1.97
	make run RELAX_FACTOR=1.98
	make run RELAX_FACTOR=1.99

exp-2-8:
	make run BORDER_EXCHANGE_FACTOR=1
	make run BORDER_EXCHANGE_FACTOR=2
	make run BORDER_EXCHANGE_FACTOR=3
	make run BORDER_EXCHANGE_FACTOR=4
	make run BORDER_EXCHANGE_FACTOR=5
	make run BORDER_EXCHANGE_FACTOR=6
	make run BORDER_EXCHANGE_FACTOR=7
	make run BORDER_EXCHANGE_FACTOR=8
	make run BORDER_EXCHANGE_FACTOR=9
	make run BORDER_EXCHANGE_FACTOR=10

exp-2-11:
	for i in 1 2 3 4 5 6 7 8 9 10 ; \
		do make run ; \
	done

.PHONY: compile run clean exp-2-2 exp-2-8 exp-2-11
