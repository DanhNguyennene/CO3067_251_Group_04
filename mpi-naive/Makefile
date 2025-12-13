CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -march=native
TARGET = mpi_program
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) mpi_program *.txt *.csv

test: $(TARGET)
	mpirun -np 4 ./$(TARGET) 100 1

test-sizes: $(TARGET)
	mpirun -np 4 ./$(TARGET) 100 1
	mpirun -np 4 ./$(TARGET) 1000 1
	mpirun -np 4 ./$(TARGET) 4000 0

test-procs: $(TARGET)
	mpirun -np 1 ./$(TARGET) 1000 0
	mpirun -np 2 ./$(TARGET) 1000 0
	mpirun -np 4 ./$(TARGET) 1000 0
	mpirun -np 8 ./$(TARGET) 1000 0

test-dnc: $(TARGET)
	mpirun -np 2 ./$(TARGET) 1000 0
	mpirun -np 4 ./$(TARGET) 2000 0
	mpirun -np 8 ./$(TARGET) 4000 0

.PHONY: all clean test test-sizes test-procs test-dnc
