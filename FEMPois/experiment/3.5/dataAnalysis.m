%% Data analysis
clear;

load timer.dat;
avg = mean(timer);
percentage_computation = avg(3)/avg(2);
percentage_exchange_borders = avg(4)/avg(2);
percentage_global_communication = avg(5)/avg(2);
percentage_idle = avg(6)/avg(2);

percentage = [percentage_computation, percentage_exchange_borders, percentage_global_communication, percentage_idle];
pie(percentage);

label = {'Computation'; 'Exchange Borders'; 'Global Communication'; 'Idle'};

legend(label);
