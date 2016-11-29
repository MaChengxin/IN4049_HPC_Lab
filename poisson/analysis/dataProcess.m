% This script reads the CSV file and extracts the data in it.

%TO DO: beautify the plot (catption, labels, autosave, etc)
%TO DO: make the script universal for all the CSV files (move to other folders and automate running)

%% Clear
clc;
clear;

%% Extract data
fileID = fopen('error_800.csv');
dataCells = textscan(fileID, 'Number of iterations: %f Error: %.6f Elapsed Wtime: %14.6f\n');
errCode = fclose(fileID);

numOfIterations = dataCells{1};
error = dataCells{2};
elapsedTime = dataCells{3};

%% Plot
semilogx(numOfIterations, error);
figure
plot(numOfIterations, elapsedTime);

%% Fitting
dimOfVector = size(numOfIterations);
numOfIterations = [ones(dimOfVector(1), 1) numOfIterations];
coefficients = mldivide(numOfIterations, elapsedTime);
alpha = coefficients(1);
beta = coefficients(2);
