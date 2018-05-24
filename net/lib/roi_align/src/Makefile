all: crop_and_resize_kernel.cu crop_and_resize_kernel.h
	 nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52



clean:
	rm crop_and_resize_kernel.cu.o

# nvcc compiler flags
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options


#/usr/local/cuda-9.1/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
 


 
