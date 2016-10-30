USERNAME=in404950
PASSWD=17ecBPwj
DAS4=fs3.das4.tudelft.nl
REMOTE_HOST=$(USERNAME)@$(DAS4)

send:
	sshpass -p $(PASSWD) scp -r helloWorld $(REMOTE_HOST):/home/$(USERNAME)

.PHONY: send