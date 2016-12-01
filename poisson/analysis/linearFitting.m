% This script reads the CSV file, extracts the data in it, and plot the
% data.

function linearFitting(filePathAndName)
%% Extract data
inputFileID = fopen(filePathAndName);
dataCells = textscan(inputFileID, 'Number of iterations: %f Error: %.6f Elapsed Wtime: %14.6f\n');
fclose(inputFileID);

numOfIterations = dataCells{1};
error = dataCells{2};
elapsedTime = dataCells{3};

%% Decompose file name
stringComponent = strsplit(filePathAndName, '/');
processTopology = char(stringComponent(4));
fileName = char(stringComponent(5));
gridSize = textscan(fileName, 'error_%d.csv');
gridSize = cell2mat(gridSize);
gridSizeStr = num2str(gridSize);

%% Plot and save figures
semilogx(numOfIterations, error);
xlabel('Number of iterations');
ylabel('Error');
title(['Error w.r.t. the number of iterations (process topology: ' processTopology ', grid size: ' gridSizeStr ')']);
legend('error');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
saveas(gcf, ['figures/error_' processTopology '_' gridSizeStr '.png']);
close;

plot(numOfIterations, elapsedTime);
xlabel('Number of iterations');
ylabel('Elapsed time');
title(['Elapsed time w.r.t. the number of iterations (process topology: ' processTopology ', grid size: ' gridSizeStr ')']);
legend('elapsed time');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
saveas(gcf, ['figures/elapsed_time_' processTopology '_' gridSizeStr '.png']);
close;

%% Fitting
dimOfVector = size(numOfIterations);
numOfIterations = [ones(dimOfVector(1), 1) numOfIterations];
coefficients = mldivide(numOfIterations, elapsedTime);
alpha = coefficients(1);
beta = coefficients(2);

%% Save result
resultFileID = fopen('result.csv', 'a');
fprintf(resultFileID, '%s\t%d\t%f\t%f\n', processTopology, gridSize, [alpha beta]);
fclose(resultFileID);
end
