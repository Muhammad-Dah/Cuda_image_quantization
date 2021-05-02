DEBUG=0

ifneq ($(DEBUG), 0)
CFLAGS=-O0 -g
else
CFLAGS=-O3 -lineinfo
endif

CFLAGS+=`pkg-config opencv --cflags --libs`

FILES=image_quantization image

all: $(FILES)

image_quantization: image_quantization.o main.o image_quantization-cpu.o
	nvcc --link $(CFLAGS) $^ -o $@

image: image_quantization-cpu.o image.o
	nvcc --link $(CFLAGS) $^ -o $@

image_quantization.o: image_quantization.cu image_quantization.h
main.o: main.cu image_quantization.h
image.o: image.cu image_quantization.h

%.o: %.cu
	nvcc --compile $< $(CFLAGS) -o $@

clean::
	rm *.o $(FILES)
