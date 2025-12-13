CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall -Wextra -march=native
LDFLAGS = -fopenmp
TARGET = main
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) main *.txt *.csv

test: $(TARGET)
	./$(TARGET) 100 1 4 128

test-sizes: $(TARGET)
	./$(TARGET) 100 1 4 128
	./$(TARGET) 1000 1 4 128
	./$(TARGET) 10000 0 4 128

test-threads: $(TARGET)
	./$(TARGET) 1000 0 1 128
	./$(TARGET) 1000 0 2 128
	./$(TARGET) 1000 0 4 128
	./$(TARGET) 1000 0 8 128
	./$(TARGET) 1000 0 16 128

test-tiled: $(TARGET)
	./$(TARGET) 1000 0 4 32
	./$(TARGET) 1000 0 4 64
	./$(TARGET) 1000 0 4 128
	./$(TARGET) 1000 0 4 256

.PHONY: all clean test test-sizes test-threads test-tiled
