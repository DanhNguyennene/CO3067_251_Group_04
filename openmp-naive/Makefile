# Makefile for OpenMP Divide & Conquer Matrix Multiplication
# Parallel Computing Assignment 1

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp

# Target executable
TARGET = main

# Source files
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)
	@echo "============================================"
	@echo "Build complete!"
	@echo "Executable: $(TARGET)"
	@echo "============================================"

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete"

# Run with default parameters (1000x1000, verify, max threads, threshold=128)
run: $(TARGET)
	@echo "Running with default parameters (1000x1000)..."
	./$(TARGET) 1000

# Quick test with small matrix and verification
test: $(TARGET)
	@echo "==================================="
	@echo "Quick Test (100x100, verify=1)"
	@echo "==================================="
	./$(TARGET) 100 1 4 64

# Test different matrix sizes (no verification for speed)
test-sizes: $(TARGET)
	@echo "==================================="
	@echo "Testing Different Matrix Sizes"
	@echo "==================================="
	@echo ""
	@echo "100x100:"
	./$(TARGET) 100 0 8 64
	@echo ""
	@echo "1000x1000:"
	./$(TARGET) 1000 0 8 128
	@echo ""
	@echo "2000x2000:"
	./$(TARGET) 2000 0 8 128

# Scalability study - varying thread counts
scalability: $(TARGET)
	@echo "==================================="
	@echo "Scalability Study (1000x1000)"
	@echo "==================================="
	@echo ""
	@echo "1 thread:"
	./$(TARGET) 1000 0 1 128
	@echo ""
	@echo "2 threads:"
	./$(TARGET) 1000 0 2 128
	@echo ""
	@echo "4 threads:"
	./$(TARGET) 1000 0 4 128
	@echo ""
	@echo "8 threads:"
	./$(TARGET) 1000 0 8 128
	@echo ""
	@echo "16 threads:"
	./$(TARGET) 1000 0 16 128

# Threshold study - testing different thresholds
threshold-study: $(TARGET)
	@echo "==================================="
	@echo "Threshold Study (1000x1000, 8 threads)"
	@echo "==================================="
	@echo ""
	@echo "Threshold = 32:"
	./$(TARGET) 1000 0 8 32
	@echo ""
	@echo "Threshold = 64:"
	./$(TARGET) 1000 0 8 64
	@echo ""
	@echo "Threshold = 128:"
	./$(TARGET) 1000 0 8 128
	@echo ""
	@echo "Threshold = 256:"
	./$(TARGET) 1000 0 8 256
	@echo ""
	@echo "Threshold = 512:"
	./$(TARGET) 1000 0 8 512

# Power-of-2 sizes (optimal for divide & conquer)
test-power2: $(TARGET)
	@echo "==================================="
	@echo "Power-of-2 Sizes (Optimal)"
	@echo "==================================="
	@echo ""
	@echo "128x128:"
	./$(TARGET) 128 0 8 64
	@echo ""
	@echo "512x512:"
	./$(TARGET) 512 0 8 128
	@echo ""
	@echo "1024x1024:"
	./$(TARGET) 1024 0 8 128
	@echo ""
	@echo "2048x2048:"
	./$(TARGET) 2048 0 8 128

# Assignment sizes (100, 1000, 10000)
assignment: $(TARGET)
	@echo "==================================="
	@echo "Assignment Sizes with Verification"
	@echo "==================================="
	@echo ""
	@echo "100x100:"
	./$(TARGET) 100 1 8 64
	@echo ""
	@echo "1000x1000:"
	./$(TARGET) 1000 1 8 128
	@echo ""
	@echo "Note: 10000x10000 verification takes very long!"
	@echo "Running 10000x10000 WITHOUT verification:"
	./$(TARGET) 10000 0 8 128

# Comprehensive benchmark for report
benchmark: $(TARGET)
	@echo "=========================================="
	@echo "Comprehensive Benchmark Suite"
	@echo "=========================================="
	@echo ""
	@echo "Part 1: Matrix Size Study"
	@echo "------------------------------------------"
	@echo "Size 100x100:"
	./$(TARGET) 100 0 8 64
	@echo ""
	@echo "Size 500x500:"
	./$(TARGET) 500 0 8 128
	@echo ""
	@echo "Size 1000x1000:"
	./$(TARGET) 1000 0 8 128
	@echo ""
	@echo "Size 2000x2000:"
	./$(TARGET) 2000 0 8 128
	@echo ""
	@echo "Size 5000x5000:"
	./$(TARGET) 5000 0 8 128
	@echo ""
	@echo "Part 2: Scalability Study (1000x1000)"
	@echo "------------------------------------------"
	@for threads in 1 2 4 8 16; do \
		echo "Threads: $$threads"; \
		./$(TARGET) 1000 0 $$threads 128; \
		echo ""; \
	done
	@echo ""
	@echo "Part 3: Threshold Study (1000x1000, 8 threads)"
	@echo "------------------------------------------"
	@for thresh in 32 64 128 256; do \
		echo "Threshold: $$thresh"; \
		./$(TARGET) 1000 0 8 $$thresh; \
		echo ""; \
	done

# Verify correctness for all assignment sizes
verify-all: $(TARGET)
	@echo "==================================="
	@echo "Correctness Verification"
	@echo "==================================="
	@echo ""
	@echo "Verifying 100x100..."
	./$(TARGET) 100 1 4 64
	@echo ""
	@echo "Verifying 500x500..."
	./$(TARGET) 500 1 4 128
	@echo ""
	@echo "Verifying 1000x1000..."
	./$(TARGET) 1000 1 8 128

# Check OpenMP support
check-openmp:
	@echo "Checking OpenMP support..."
	@echo "#include <omp.h>" > check_omp.cpp
	@echo "#include <iostream>" >> check_omp.cpp
	@echo "int main() { std::cout << \"Max threads: \" << omp_get_max_threads() << std::endl; return 0; }" >> check_omp.cpp
	@$(CXX) -fopenmp check_omp.cpp -o check_omp 2>/dev/null && \
		(./check_omp && echo "✓ OpenMP is supported") || \
		echo "✗ OpenMP is NOT supported"
	@rm -f check_omp.cpp check_omp

# Show system information
info:
	@echo "=========================================="
	@echo "System Information"
	@echo "=========================================="
	@echo "Compiler: $(CXX)"
	@$(CXX) --version | head -n 1
	@echo ""
	@echo "CPU Info:"
	@if [ -f /proc/cpuinfo ]; then \
		echo "  Model: $$(grep 'model name' /proc/cpuinfo | head -n 1 | cut -d: -f2 | sed 's/^[ \t]*//')"; \
		echo "  Cores: $$(grep -c ^processor /proc/cpuinfo)"; \
	else \
		echo "  (CPU info not available)"; \
	fi
	@echo ""
	@echo "OpenMP Info:"
	@make check-openmp 2>/dev/null || echo "  Unable to check OpenMP"

# Help message
help:
	@echo "=========================================="
	@echo "OpenMP Divide & Conquer Matrix Multiplication"
	@echo "=========================================="
	@echo ""
	@echo "Build targets:"
	@echo "  make              - Build the program"
	@echo "  make clean        - Remove build files"
	@echo ""
	@echo "Testing targets:"
	@echo "  make test         - Quick test (100x100 with verification)"
	@echo "  make run          - Run with defaults (1000x1000)"
	@echo "  make test-sizes   - Test multiple sizes"
	@echo "  make test-power2  - Test power-of-2 sizes (optimal)"
	@echo "  make assignment   - Run assignment sizes (100, 1000, 10000)"
	@echo ""
	@echo "Analysis targets:"
	@echo "  make scalability  - Thread count study"
	@echo "  make threshold-study - Threshold optimization study"
	@echo "  make benchmark    - Comprehensive benchmark suite"
	@echo "  make verify-all   - Verify correctness for all sizes"
	@echo ""
	@echo "Utility targets:"
	@echo "  make check-openmp - Check if OpenMP is available"
	@echo "  make info         - Show system information"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Manual execution:"
	@echo "  ./$(TARGET) <size> [verify] [threads] [threshold]"
	@echo ""
	@echo "  Arguments:"
	@echo "    size      - Matrix dimension (e.g., 1000)"
	@echo "    verify    - 0=skip verification, 1=verify (default: 1)"
	@echo "    threads   - Number of OpenMP threads (default: max available)"
	@echo "    threshold - Base case size (default: 128)"
	@echo ""
	@echo "  Examples:"
	@echo "    ./$(TARGET) 1000                  # 1000x1000, verify, max threads, threshold=128"
	@echo "    ./$(TARGET) 2000 0                # 2000x2000, no verify"
	@echo "    ./$(TARGET) 1000 1 8              # 1000x1000, verify, 8 threads"
	@echo "    ./$(TARGET) 1000 0 16 64          # 1000x1000, no verify, 16 threads, threshold=64"
	@echo ""
	@echo "⚠️  Note: Non-power-of-2 sizes will fall back to naive multiplication"
	@echo "          for odd-sized blocks, reducing parallelism."

.PHONY: all clean run test test-sizes scalability threshold-study test-power2 \
        assignment benchmark verify-all check-openmp info help