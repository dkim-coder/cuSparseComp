#include <iostream>
#include <cstdlib>
#include <cmath>
#include<time.h>
#include "_kernel.cuh"


template <typename T>
void printMatrix(const T* matrix, const int row, const int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << matrix[col * i + j] << " ";
        }
        std::cout << std::endl;
    }

    return;
}


template <typename T = __half>
void compareMatrix(const T* A, const T* B, const int row, const int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            auto pos = col * i + j;
            if (A[pos] != B[pos]) {
                std::cout << "The tow matrices are different." << std::endl;
                return;
            }
        }
    }

    return;
}

template <typename T = __half>
void makeSparsity(T* A, const int row, const int col, const double sparsity)
{
    int target_nz = static_cast<int>(ceil(row * col * sparsity));     // number of zero in matrix 
    int matrix_nz = 0;
    int i = 0;
    int j = 0;
    int pos;

    // find nubmer of zero in matrix
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            if (A[col * i + j] == 0) matrix_nz++;
        }
    }   

    bool exitOuterLoop = false;
    srand(time(NULL));
    while (true) {
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j += (rand() % 4)) {
                pos = col * i + j;
                if (A[pos] != 0) {
                    A[pos] = 0.;
                    matrix_nz += 1;
                    if (matrix_nz == target_nz) {
                        exitOuterLoop = true;
                        break;
                    }
                }
                if (exitOuterLoop == true)break;
            }
            if (exitOuterLoop == true)break;
        }
        if (exitOuterLoop == true)break;
    }    

    return;
}


int main(void)
{
    // A(m * k), B(k, n), C(m, n) --> row-major order
    int m, n, k;
    std::cout << "input m, n, k" << std::endl;
    std::cin >> m; std::cin >> n; std::cin >> k;

    auto A_size = m * k;
    auto B_size = k * n;
    auto C_size = m * n;

    // host memory allocation
    __half* hA = (__half*)malloc(A_size * sizeof(__half));
    __half* hA_pruned = (__half*)malloc(A_size * sizeof(__half));   memset(hA_pruned, 0, A_size * sizeof(__half));
    __half* hB = (__half*)malloc(B_size * sizeof(__half));
    __half* lt_hC = (__half*)malloc(C_size * sizeof(__half));           memset(lt_hC, 0, C_size * sizeof(__half));
    __half* coo_hC = (__half*)malloc(C_size * sizeof(__half));           memset(coo_hC, 0, C_size * sizeof(__half));
    __half* csr_hC = (__half*)malloc(C_size * sizeof(__half));           memset(csr_hC, 0, C_size * sizeof(__half));
    __half* csc_hC = (__half*)malloc(C_size * sizeof(__half));           memset(csc_hC, 0, C_size * sizeof(__half));


    srand(time(NULL));
    // fill matrix A, matrix B  0 ~ 9 
    for (int i = 0; i < A_size; i++)
        hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
    for (int i = 0; i < B_size; i++)
        hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));

    
    // sparsity 50%
    std::cout << "sparisty is 50%" << std::endl;
    cusLtMatmul(hA, hB, lt_hC, hA_pruned, m, n, k);     // --> hA_pruned copy ÇØ¿Â´Ù.
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k);    
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k);
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k);
    
    // sparsity 75%
    std::cout << "sparisty is 75%" << std::endl;
    makeSparsity(hA_pruned, m, k, 0.75);
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k);
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k);
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k);
    
    // sparsity 87.5%
    std::cout << "sparisty is 87.5%" << std::endl;
    makeSparsity(hA_pruned, m, k, 0.875);
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k);
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k);
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k);
    
    // sparsity 99%
    std::cout << "sparisty is 99%" << std::endl;
    makeSparsity(hA_pruned, m, k, 0.99);
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k);
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k);
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k);


    // compareMatrix(lt_hC, csc_hC, m, n);


    // free host memory
    free(hA);
    free(hA_pruned);
    free(hB);
    free(lt_hC);
    free(coo_hC);
    free(csr_hC);
    free(csc_hC);

    return EXIT_SUCCESS;
}