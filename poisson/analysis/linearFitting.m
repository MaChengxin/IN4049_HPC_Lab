% This script reads the CSV file and extracts the data in it.

%TO DO: beautify the plot (catption, labels, autosave, etc)

function linearFitting(filePathAndName)
close all;

%% Extract data
inputFileID = fopen(filePathAndName);
dataCells = textscan(inputFileID, 'Number of iterations: %f Error: %.6f Elapsed Wtime: %14.6f\n');
fclose(inputFileID);
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

%% Save result
resultFileID = fopen('result.csv', 'a');
stringComponent = strsplit(filePathAndName, '/');
processTopology = char(stringComponent(4));
fileName = char(stringComponent(5));
gridSize = textscan(fileName, 'error_%d.csv');
gridSize = cell2mat(gridSize);
fprintf(resultFileID, '%s\t%d\t%f\t%f\n', processTopology, gridSize, [alpha beta]);
fclose(resultFileID);
end
