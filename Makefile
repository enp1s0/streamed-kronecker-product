NVCC=nvcc
NVCCFLAGS=-std=c++11 -lcublas -arch=sm_60

all:batched-k seq-k


batched-k:main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -DBATCHED

seq-k:main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm seq-k batched-k
