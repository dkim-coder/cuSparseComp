#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

template <typename T> int cusLtMatmul(T* hA, T* hB, T* hC, T* hA_pruned, const int m, const int n, const int k, const int device_num);
template <typename T> int cusMatmulCoo(T* hA_pruned, T* hB, T* hC, const int m, const int n, const int k);
template <typename T> int cusMatmulCsr(T* hA_pruned, T* hB, T* hC, const int m, const int n, const int k);
template <typename T> int cusMatmulCsc(T* hA_pruned, T* hB, T* hC, const int m, const int n, const int k);
#include "kernel.cu"

#endif