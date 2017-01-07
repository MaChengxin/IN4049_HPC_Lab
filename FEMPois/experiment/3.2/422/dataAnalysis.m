%% Data analysis
clear;

load timer_100.dat;
avg = mean(timer_100);
percentage_computation = avg(3)/avg(2);
percentage_exchange_borders = avg(4)/avg(2);
percentage_global_communication = avg(5)/avg(2);
percentage_idle = avg(6)/avg(2);

percentage = [percentage_computation, percentage_exchange_borders, percentage_global_communication, percentage_idle];
h = pie(percentage);

txt = {'Computation: '; 'Exchange Borders: '; 'Global Communication: '; 'Idle: '};
hText = findobj(h, 'Type', 'text');
percentValues = get(hText, 'String');
combinedtxt = strcat(txt, percentValues);
hText(1).String = combinedtxt(1);
hText(2).String = combinedtxt(2);
hText(3).String = combinedtxt(3);
hText(4).String = combinedtxt(4);
