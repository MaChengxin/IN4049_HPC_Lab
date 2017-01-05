CC=gcc
SRC=SEQ_Poisson.c
EXEC=SEQ_Poisson

compile:
	$(CC) -o $(EXEC) $(SRC)

run: compile
	./$(EXEC)

clean:
	rm -f output.dat $(EXEC)	

.PHONY: compile run clean
