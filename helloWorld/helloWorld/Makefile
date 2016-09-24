CC=mpicc
SRC=helloWorld.c
EXEC=helloWorld

NODESNUM=4
CORESNUM=4

compile:
	$(CC) -o $(EXEC) $(SRC)

run:
	prun -v -np $(NODESNUM) -$(CORESNUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC)

.PHONY: compile run