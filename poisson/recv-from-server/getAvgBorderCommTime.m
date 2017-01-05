%% This file reads border_exchange_info*.dat and processes its data.

function avgTime = getAvgBorderCommTime(filePathAndName)
%% Extract data
inputFileID = fopen(filePathAndName);
dataCells = textscan(inputFileID, '%d %f\n');
fclose(inputFileID);

timePerProc = dataCells{2};
avgTime = mean(timePerProc);

end
