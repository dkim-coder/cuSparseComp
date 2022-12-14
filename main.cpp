#include <iostream>
#include <cstdlib>
#include <cmath>
#include<time.h>
#include "_kernel.cuh"
#include "main.h"
#include<algorithm>

typedef float input_type;

#define HALF_EPSILON 0.001
#define FLOAT_EPSILON 0.000001

template <typename T>
void printMatrix(const T* matrix, const int row, const int col)
{
    int pos;
    int cnt = 0;  
    int r_cnt = 0;

    std::cout << "\n-------------------------------------------------------------------------------" << std::endl;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            pos = col * i + j;
            if (matrix[pos] == 0) {
                cnt++;
                r_cnt++;
            }
            std::cout << matrix[pos] << " ";
        }
        std::cout << "--> " << r_cnt << std::endl;
        r_cnt = 0;
    }
    std::cout << "-------------------------------------------------------------------------------\n" << std::endl;
    std::cout << "total number of zero is : " << cnt << std::endl;

    return;
}

template <typename T>
void compareMatrix(const T* A, const T* B, const int row, const int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            auto pos = col * i + j;
            if (A[pos] - B[pos] > FLOAT_EPSILON) {     // precision EPSILON
                std::cout << "Two matrices are different." << std::endl;
                return;
            }
        }
    }
    std::cout << "Two matrices are same." << std::endl;

    return;
}

template <typename T>
void makeSparsity(T* A, const int row, const int col, const double sparsity)
{
    int target_nz = static_cast<int>(ceil(row * col * sparsity)); 
    int matrix_nz = 0;  // number of zero in matrix
    int i, j;
    int pos;

    // find nubmer of zero in matrix
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            if (A[col * i + j] == 0) matrix_nz++;
        }
    }   

    // make sparsity
    int p;
    bool exitOuterLoop = false;
    srand(time(NULL));
    while (true) {
        for (i = 0; i < row; i++) {
            p = 0;
            for (j = rand() % 4; j < col; j = p * 4 + (rand() % 4)) {
                p++;
                pos = col * i + j;
                if (A[pos] != 0) {
                    A[pos] = static_cast<T>(0.f);
                    matrix_nz++;
                    if (matrix_nz == target_nz) {
                        exitOuterLoop = true;
                        break;
                    }
                }
            }
            if (exitOuterLoop == true)break;
        }
        if (exitOuterLoop == true)break;
    }
    std::cout << "\nA_pruned Sparsity is : " << (double)target_nz / (row * col) << std::endl << std::endl;

    return;
}


// upgrade makeSparsity function
template <typename T>
void makeSparsity2(T* A, const int row, const int col, const double sparsity)
{
    int target_nz = static_cast<int>(ceil(row * col * sparsity));
    int matrix_nz = 0;  // number of zero in matrix
    int i, j;
    int row_nz = static_cast<int>(floor(col * sparsity));
    int* each_row_nz = (int*)malloc(row * sizeof(int)); memset(each_row_nz, 0, row * sizeof(int));
    int pos;

    // find nubmer of zero in matrix
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            if (A[col * i + j] == 0) {
                matrix_nz++;
                each_row_nz[i]++;    // ?? 0 ???? ?? update
            }
        }
    }

    int limit = row_nz * row;
    srand(time(NULL));
    while (matrix_nz < target_nz) {
        for (i = 0; i < row; i++) {
            if (limit == matrix_nz) break;
            if ((each_row_nz[i] < row_nz) && (matrix_nz < target_nz)) {
                while (true) {
                    pos = col * i + (std::rand() % col);
                    if (A[pos] != 0) {
                        A[pos] = 0;
                        matrix_nz++;
                        each_row_nz[i]++;
                        break;
                    }
                }
            }
        }
        if (limit == matrix_nz) break;
    }
    while (matrix_nz < target_nz) {
        for (i = 0; i < row; i++) {
            if ((each_row_nz[i] <= row_nz) && (matrix_nz < target_nz)) {
                while (true) {
                    pos = col * i + (std::rand() % col);
                    if (A[pos] != 0) {
                        A[pos] = 0;
                        matrix_nz++;
                        each_row_nz[i]++;
                        break;
                    }
                }
            }
        }
    }
    std::cout << "\nA_pruned Sparsity is : " << (double)target_nz / (row * col) << std::endl << std::endl;
    
    // std::cout << "\n max :" << * std:: max_element(each_row_nz, each_row_nz + row) << " min : " << *std:: min_element(each_row_nz, each_row_nz + row) << std::endl;
    free(each_row_nz);

    return;
}

// A(m x k), B(k x n), C(m x n) --> row-major order
int main(void)
{

    // input m, n, k
    int m, n, k;
    std::cout << "input m, n, k " << std::endl;
    std::cin >> m; std::cin >> n; std::cin >> k;

    auto A_size = m * k;
    auto B_size = k * n;
    auto C_size = m * n;

    // host memory allocation
    input_type* hA = (input_type*)malloc(A_size * sizeof(input_type));
    input_type* hA_pruned = (input_type*)malloc(A_size * sizeof(input_type));   memset(hA_pruned, 0, A_size * sizeof(input_type));
    input_type* hB = (input_type*)malloc(B_size * sizeof(input_type));
    input_type* lt_hC = (input_type*)malloc(C_size * sizeof(input_type));           memset(lt_hC, 0, C_size * sizeof(input_type));
    input_type* coo_hC = (input_type*)malloc(C_size * sizeof(input_type));           memset(coo_hC, 0, C_size * sizeof(input_type));
    input_type* csr_hC = (input_type*)malloc(C_size * sizeof(input_type));           memset(csr_hC, 0, C_size * sizeof(input_type));
    input_type* csc_hC = (input_type*)malloc(C_size * sizeof(input_type));           memset(csc_hC, 0, C_size * sizeof(input_type));


    // fill matrix A, matrix B  0 ~ 9
    srand(time(NULL));
    for (int i = 0; i < A_size; i++)
        hA[i] = static_cast<float>(std::rand() % 10);
    for (int i = 0; i < B_size; i++)
        hB[i] = static_cast<float>(std::rand() % 10);
    for (int i = 0; i < A_size; i++)
        hA_pruned[i] = static_cast<float>(std::rand() % 10);

    // sparsity 50%
    std::cout << "\n---------- sparisty is 50% ----------" << std::endl;
    cusLtMatmul(hA, hB, lt_hC, hA_pruned, m, n, k);    // --> hA_pruned copy ??????.      //  16?????? ???? 8?????? ??????.
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k); 
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k); 
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k); 


    // sparsity 75%
    std::cout << "\n---------- sparisty is 75% ----------" << std::endl;
    makeSparsity2(hA_pruned, m, k, 0.75);
    memset(coo_hC, 0, C_size * sizeof(input_type));
    memset(csr_hC, 0, C_size * sizeof(input_type));
    memset(csc_hC, 0, C_size * sizeof(input_type));
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k); 
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k); 
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k);


    // sparsity 87.5%
    std::cout << "\n---------- sparisty is 87.5% ----------" << std::endl;
    makeSparsity2(hA_pruned, m, k, 0.875);
    memset(coo_hC, 0, C_size * sizeof(input_type));
    memset(csr_hC, 0, C_size * sizeof(input_type));
    memset(csc_hC, 0, C_size * sizeof(input_type));
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k); 
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k); 
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k);


    // sparsity 99%
    std::cout << "\n---------- sparisty is 99% ----------" << std::endl;
    makeSparsity2(hA_pruned, m, k, 0.99);
    memset(coo_hC, 0, C_size * sizeof(input_type));
    memset(csr_hC, 0, C_size * sizeof(input_type));
    memset(csc_hC, 0, C_size * sizeof(input_type));
    cusMatmulCoo(hA_pruned, hB, coo_hC, m, n, k); 
    cusMatmulCsr(hA_pruned, hB, csr_hC, m, n, k); 
    cusMatmulCsc(hA_pruned, hB, csc_hC, m, n, k); 


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