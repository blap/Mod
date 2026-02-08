CC = gcc
CFLAGS = -fPIC -shared -O3 -Wall -Wextra
TARGET = libtensor_ops.so
SRC = tensor_ops.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f $(TARGET)
