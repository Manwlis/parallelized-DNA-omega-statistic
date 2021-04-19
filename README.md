# parallelized-DNA-omega-statistic

Parallelize a simple program using SSE and mpi. Use the script to compile and run all the files. Needs lamboot and MPICH to run.

The program computes a simplified version of ω statistic for N random data and prints the greatest value. Also, it times the calculations. 

Benchmarked on dual Intel Xeons E5620 @ 2.40GHz with 8GB DDR3 memory.


## Parallelization

Two different implementations have been requested, first using only SSE and second using both SSE and mpi. Also the unroll and jam versions of the serial code have been added.


## Script

Executes all versions with 4 different data sizes. Also, runs mpi with 2, 4 and 8 processes.
