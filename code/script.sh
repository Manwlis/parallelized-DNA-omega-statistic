#!/bin/bash
# Description :	Compile and run all the variations of the algorithm witn the same inputs.

echo ""
echo "The script starts."
echo ""

#Compile the code.
gcc -o serial serial.c
gcc -o SSE SSE.c -msse4.2
mpicc -o MPI_SSE MPI_SSE.c -msse4.2 -lm


#Run the code.

echo "Serial executions."
echo ""

./serial '100'
./serial '1000'
./serial '10000'
./serial '100000'

echo ""
echo "SSE executions."
echo ""

./SSE '100'
./SSE '1000'
./SSE '10000'
./SSE '100000'

echo ""

lamboot -v host

echo ""
echo "MPI + SSE executions."
echo ""
echo "2 processes."
echo ""

mpiexec -n 2 ./MPI_SSE '100'
mpiexec -n 2 ./MPI_SSE '1000'
mpiexec -n 2 ./MPI_SSE '10000'
mpiexec -n 2 ./MPI_SSE '100000'

echo ""
echo "4 processes."
echo ""

mpiexec -n 4 ./MPI_SSE '100'
mpiexec -n 4 ./MPI_SSE '1000'
mpiexec -n 4 ./MPI_SSE '10000'
mpiexec -n 4 ./MPI_SSE '100000'
echo ""
echo "8 processes."
echo ""

mpiexec -n 8 ./MPI_SSE '100'
mpiexec -n 8 ./MPI_SSE '1000'
mpiexec -n 8 ./MPI_SSE '10000'
mpiexec -n 8 ./MPI_SSE '100000'

lamhalt