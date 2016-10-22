CC=gcc
SRC=SEQ_Poisson.c
EXEC=SEQ_Poisson

compile:
	$(CC) -o $(EXEC) $(SRC)

run:
	./$(EXEC)

Doxyfile:
	doxygen -g

callgraph:
	doxygen Doxyfile
	
.PHONY: compile run Doxyfile callgraph
