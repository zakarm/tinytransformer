CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -I include -I .

SRC_DIR  := src
BUILD_DIR := build
LIB      := libtinytransformer.a

SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

MAIN_SRC := main.cpp
TARGET   := tinytransformer

.PHONY: all lib clean

all: lib $(if $(wildcard $(MAIN_SRC)), $(TARGET))

lib: $(BUILD_DIR) $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(TARGET): $(MAIN_SRC) $(LIB)
	$(CXX) $(CXXFLAGS) $< -L. -ltinytransformer -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(LIB) $(TARGET)
