# Makefile for CPU Ops Plugin (Linux/Windows)

# Detect OS
ifeq ($(OS),Windows_NT)
    TARGET_LIB = libtensor_ops.dll
    CC = gcc
    CFLAGS = -shared -O3 -Wall -Wextra -fopenmp -D_WIN32
    LDFLAGS = -lm
    RM = del
else
    TARGET_LIB = libtensor_ops.so
    CC = gcc
    CFLAGS = -fPIC -shared -O3 -Wall -Wextra -fopenmp
    LDFLAGS = -lm
    RM = rm -f
endif

SRC = tensor_ops.c safetensors_loader.c
OBJ = $(SRC:.c=.o)

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) $(TARGET_LIB) *.o
