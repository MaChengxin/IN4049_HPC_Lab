CC = mpicc

NODES_NUM = 1
CORES_NUM = 4

FP_LIBS = -lm
GD_LIBS = -lm

FP_OBJS = MPI_Fempois.o
GD_OBJS = GridDist.o

all: MPI_Fempois GridDist

clean:
	rm -f *.o 

MPI_Fempois: $(FP_OBJS)
	mpicc -o $@ $(FP_OBJS) $(FP_LIBS)

GridDist: $(GD_OBJS)
	gcc -o $@ $(GD_OBJS) $(GD_LIBS)

MPI_Fempois.o:
	mpicc -c MPI_Fempois.c

GridDist.o:
	gcc -c GridDist.c

run:
	prun -v -np $(NODES_NUM) -$(CORES_NUM) -sge-script $$PRUN_ETC/prun-openmpi ./MPI_Fempois

.PHONY: run
