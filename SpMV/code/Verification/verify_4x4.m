%% This code verifies the result of the CUDA application.

clear

A = [1 7 0 0;
    0 2 8 0;
    5 0 3 9;
    0 6 0 4];

y = [1 2 3 4];
y = y';

x = [1 2 3 4];
x = x';

y = y + A * x
