#include <iostream>
#include <cstdlib>
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
    __half* csr_hC = (__half*)malloc(C_size * sizeof(__half));           memset(csr_hC, 0, C_size * sizeof(__half));
    __half* coo_hC = (__half*)malloc(C_size * sizeof(__half));           memset(coo_hC, 0, C_size * sizeof(__half));

    // fill matrix A, matrix B
    for (int i = 0; i < A_size; i++)
        hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
    for (int i = 0; i < B_size; i++)
        hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));


    cusLtMatmul(hA, hB, lt_hC, hA_pruned, m, n, k, 0);     // --> hA_pruned copy ÇØ¿Â´Ù.

    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k);

    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k);

    // compareMatrix(lt_hC, coo_hC, m, n);


    // free host memory
    free(hA);
    free(hB);
    free(lt_hC);
    free(coo_hC);
    free(csr_hC);
    free(hA_pruned);

    return EXIT_SUCCESS;
}