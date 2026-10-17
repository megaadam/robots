CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

TARGET := hello
SRC := hello.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
