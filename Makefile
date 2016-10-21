DAS4=in404950@fs3.das4.tudelft.nl
PASSWD=17ecBPwj

conn:
	sshpass -p $(PASSWD) ssh $(DAS4)

.PHONY: conn
