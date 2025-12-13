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
	mpirun -np 7 ./$(TARGET) 100 1

test-sizes: $(TARGET)
	mpirun -np 7 ./$(TARGET) 100 1
	mpirun -np 7 ./$(TARGET) 1000 1
	mpirun -np 7 ./$(TARGET) 4000 0

.PHONY: all clean test test-sizes
