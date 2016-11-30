% Automate running for results

delete result.csv;
clear;
clc;

resultFileID = fopen('result.csv', 'a');
fprintf(resultFileID, '%s\t%s\t%s\t%s\n', 'Process Topology', 'Grid Size', 'Alpha', 'Beta');
fclose(resultFileID);

linearFitting('../experiment/2.3/4x1/error_200.csv');
linearFitting('../experiment/2.3/4x1/error_400.csv');
linearFitting('../experiment/2.3/4x1/error_800.csv');
linearFitting('../experiment/2.3/2x2/error_200.csv');
linearFitting('../experiment/2.3/2x2/error_400.csv');
linearFitting('../experiment/2.3/2x2/error_800.csv');

close all;
