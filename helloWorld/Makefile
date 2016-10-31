CC=mpicc
SRC=helloWorld.c
EXEC=helloWorld

NODESNUM=4
CORESNUM=2

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	prun -v -np $(NODESNUM) -$(CORESNUM) -sge-script $$PRUN_ETC/prun-openmpi ./$(EXEC)

clean:
	rm -f $(EXEC)

.PHONY: compile run clean
