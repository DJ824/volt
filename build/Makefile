CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pthread

PROJECT_ROOT := ..
TARGET := volt
SOURCES := $(PROJECT_ROOT)/main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) $(PROJECT_ROOT)/include/thread_pool.h $(PROJECT_ROOT)/include/work_steal_deque.h $(PROJECT_ROOT)/include/mpsc_seq.h $(PROJECT_ROOT)/include/spsc_final.h
	$(CXX) $(CXXFLAGS) -I$(PROJECT_ROOT) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
