#include <cusparseLt.h>

#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

int cusLtMatmul(__half* hA, __half* hB, __half* hC, __half* hA_pruned, const int m, const int n, const int k, const int device_num);
int cusMatmulCoo(__half* hA_pruned, __half* hB, __half* hC, const int m, const int n, const int k);
int cusMatmulCsr(__half* hA_pruned, __half* hB, __half* hC, const int m, const int n, const int k);
int cusMatmulCsc(__half* hA_pruned, __half* hB, __half* hC, const int m, const int n, const int k);

#endif