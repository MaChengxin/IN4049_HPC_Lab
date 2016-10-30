CC=gcc
SRC=SEQ_Poisson.c
EXEC=SEQ_Poisson

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	./$(EXEC)

Doxyfile:
	doxygen -g

callgraph:
	doxygen Doxyfile

clean:
	rm -f output.dat $(EXEC)	

.PHONY: compile run Doxyfile callgraph clean
