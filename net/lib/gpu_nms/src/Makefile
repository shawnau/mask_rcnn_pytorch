all: gpu_nms_kernel.cu
	 nvcc -c -o gpu_nms_kernel.cu.o gpu_nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52


clean:
	rm gpu_nms_kernel.cu.o
 

#/usr/local/cuda-9.1/bin/nvcc -c -o gpu_nms_kernel.cu.o gpu_nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
 

