@echo off
gcc -g -Wall -Wextra *.c -O3 -flto -o regression
