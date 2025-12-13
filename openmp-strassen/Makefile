CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall -Wextra -march=native
LDFLAGS = -fopenmp
TARGET = optimized_main
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) optimized_main *.txt *.csv

test: $(TARGET)
	./$(TARGET) 100 1 7 128

test-sizes: $(TARGET)
	./$(TARGET) 100 1 7 128
	./$(TARGET) 1000 1 7 128
	./$(TARGET) 10000 0 7 128

test-threads: $(TARGET)
	./$(TARGET) 1000 0 7 128
	./$(TARGET) 1000 0 14 128
	./$(TARGET) 1000 0 21 128
	./$(TARGET) 1000 0 28 128

.PHONY: all clean test test-sizes test-threads