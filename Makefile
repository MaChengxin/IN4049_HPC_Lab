DAS4=in404950@fs3.das4.tudelft.nl

conn:
	ssh $(DAS4)
send:
	scp helloWorld.c $(DAS4):helloWorld
