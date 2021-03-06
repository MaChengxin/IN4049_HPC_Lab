USERNAME=in404950
PASSWD=17ecBPwj
DAS4=fs3.das4.tudelft.nl
REMOTE_HOST=$(USERNAME)@$(DAS4)

conn:
	sshpass -p $(PASSWD) ssh $(REMOTE_HOST)

send-hello-world:
	sshpass -p $(PASSWD) scp -r helloWorld $(REMOTE_HOST):/home/$(USERNAME)

send-poisson-par:
	sshpass -p $(PASSWD) scp -r poisson/code/par $(REMOTE_HOST):/home/$(USERNAME)/poisson

send-fempois:
	sshpass -p $(PASSWD) scp -r FEMPois/code $(REMOTE_HOST):/home/$(USERNAME)/FEMPois

fetch-poisson-profile:
	sshpass -p $(PASSWD) scp $(REMOTE_HOST):/home/$(USERNAME)/poisson/profile*.csv poisson/recv-from-server

fetch-poisson-error:
	sshpass -p $(PASSWD) scp $(REMOTE_HOST):/home/$(USERNAME)/poisson/error*.csv poisson/recv-from-server

fetch-poisson-border-info:
	sshpass -p $(PASSWD) scp $(REMOTE_HOST):/home/$(USERNAME)/poisson/border*.dat poisson/recv-from-server

fetch-fempois-timer:
	sshpass -p $(PASSWD) scp $(REMOTE_HOST):/home/$(USERNAME)/FEMPois/timer.dat FEMPois/recv-from-server

.PHONY: conn send-hello-world send-poisson-par send-fempois fetch-poisson-profile fetch-poisson-error fetch-poisson-border-info fetch-fempois-timer
