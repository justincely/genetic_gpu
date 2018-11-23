all: assignment.cu
	nvcc assignment.cu -o assignment.exe -lcudart
