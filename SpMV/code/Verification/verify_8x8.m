%% This code verifies the result of the CUDA application.

clear

A = [1 0 2 3 0 0 4 5;
    0 1 0 2 0 0 0 0;
    0 0 0 0 0 0 0 0;
    1 2 3 4 5 0 6 7;
    0 1 0 2 0 3 0 0;
    1 2 0 0 0 0 0 0;
    0 1 2 3 4 5 6 7;
    1 2 3 4 5 6 7 8];

y = [1 2 3 4 5 6 7 8];
y = y';

x = [1 2 3 4 5 6 7 8];
x = x';

y = y + A * x
