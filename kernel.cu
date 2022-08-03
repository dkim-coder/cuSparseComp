#include <iostream>
#include <cuda_runtime_api.h>      // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>            // cusparseLt header  --> #include <cusparse.h>

                                                                 
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}                                                                               

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}  


constexpr int EXIT_UNSUPPORTED = 2;

// cusparseLtMatmul  -->  D = alpha * A * B + beta * C
template <typename T>
int cusLtMatmul(T* hA, T* hB, T* hC, T* hA_pruned, const int m, const int n, const int k, const int device_num = 0)
{
    int major_cc, minor_cc;
    CHECK_CUDA(cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, device_num))
    CHECK_CUDA(cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, device_num))
    if (!(major_cc == 8 && minor_cc == 0) && !(major_cc == 8 && minor_cc == 6)) {
        std::cout << "\ncusparseLt is supported only on GPU devices with compute capability == 8.0, 8.6 current: " << major_cc << "." << minor_cc << std::endl;
        return EXIT_UNSUPPORTED;
    }
    
    // matrix A, B, C : NON_TRANSPOSE, row-major
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type = CUDA_R_32F;
    auto          compute_type = CUSPARSE_COMPUTE_TF32;
    bool     is_rowmajor = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows = (isA_transposed) ? k : m;
    auto     num_A_cols = (isA_transposed) ? m : k;
    auto     num_B_rows = (isB_transposed) ? n : k;
    auto     num_B_cols = (isB_transposed) ? k : n;
    auto     num_C_rows = m;
    auto     num_C_cols = n;
    unsigned alignment = 32;    // mutiple of 32   (processor가 8byte단위로 읽어옴 64bit os) --> 프로세서 접근 횟수 줄여줌
    auto     lda = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size = A_height * lda * sizeof(T);
    auto     B_size = B_height * ldb * sizeof(T);
    auto     C_size = C_height * ldc * sizeof(T);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Device memory management
    T* dA, * dB, * dC, * dD, * dA_compressed;
    CHECK_CUDA(cudaMalloc((void**)&dA, A_size))  
    CHECK_CUDA(cudaMalloc((void**)&dB, B_size))  
    CHECK_CUDA(cudaMalloc((void**)&dC, C_size))  
    dD = dC;

    CHECK_CUDA(cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice))  
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice))  
    CHECK_CUDA(cudaMemset(dC, 0, C_size))   

    // ---------------------------------------------------------------------------------------------
    cusparseLtHandle_t             handle;  
    cusparseLtMatDescriptor_t      matA, matB, matC;    
    cusparseLtMatmulDescriptor_t   matmul;  
    cusparseLtMatmulAlgSelection_t alg_sel; 
    cusparseLtMatmulPlan_t         plan;     
    cudaStream_t                   stream = nullptr;
    
    CHECK_CUSPARSE(cusparseLtInit(&handle))
    // matrix descriptor initialization
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, num_A_rows, num_A_cols, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT)) 
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, num_B_rows, num_B_cols, ldb, alignment, type, order))  
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, num_C_rows, num_C_cols, ldc, alignment, type, order))  
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))    
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT)) 
    int alg = 0;    // algorithm attribute number
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))   
    size_t workspace_size;
    workspace_size = 10000;
    //CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size)) 
    // Prune the A matrix (in-place) and check the correcteness
    int* d_valid;
    CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(d_valid)))
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream)) 
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream))    
    int is_valid;
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(*d_valid), cudaMemcpyDeviceToHost, stream))
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (is_valid != 0) {
        std::cout << "!!!! The matrix has been pruned in a wrong way. cusparseLtMatmul will not provide correct results" << std::endl;
        return EXIT_FAILURE;
    }

    size_t compressed_size;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size)) 
    CHECK_CUDA(cudaMalloc((void**)&dA_compressed, compressed_size)) // memory allocation dA_compressed
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream))  

    
    void* d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams = nullptr;
    /* CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams, num_streams))   // ------> 문제 부분
         int alg_id;
         CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)))
         std::cout << "alg_id : " << alg_id << std::endl; */

    // Perform the matrix multiplication
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams, num_streams))
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cusparseLt spending time : " << ms << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // memory copy device to host
    CHECK_CUDA(cudaMemcpy(hA_pruned, dA, A_size, cudaMemcpyDeviceToHost))   // pruned with 50sparsity matrix A copy to host 
    CHECK_CUDA(cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost))  

    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    
    // device memory deallocation
    CHECK_CUDA(cudaFree(dA_compressed))
    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    CHECK_CUDA(cudaFree(d_valid))

    return EXIT_SUCCESS;
}

template <typename T>
int cusMatmulCoo(T* hA_pruned, T* hB, T* hC, const int m, const int n, const int k)
{
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type = CUDA_R_32F;
    auto          compute_type = CUDA_R_32F;
    bool     is_rowmajor = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows = (isA_transposed) ? k : m;
    auto     num_A_cols = (isA_transposed) ? m : k;
    auto     num_B_rows = (isB_transposed) ? n : k;
    auto     num_B_cols = (isB_transposed) ? k : n;
    auto     num_C_rows = m;
    auto     num_C_cols = n;
    auto     lda = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size = A_height * lda * sizeof(T);
    auto     B_size = B_height * ldb * sizeof(T);
    auto     C_size = C_height * ldc * sizeof(T);
            
    float alpha = 1.0f;
    float beta = 0.0f;
  
    // Device memory management
    T * dA_pruned, * dB, * dC;
    CHECK_CUDA(cudaMalloc((void**)&dA_pruned, A_size))
    CHECK_CUDA(cudaMalloc((void**)&dB, B_size))
    CHECK_CUDA(cudaMalloc((void**)&dC, C_size))
    CHECK_CUDA(cudaMemcpy(dA_pruned, hA_pruned, A_size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))   
    
    int* d_coo_rows, * d_coo_columns;
    T* d_coo_values;
        
    // convert dense matrix tmpA --> sparse matrix matA in COO format
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t tmpA, matB, matC;
    void* dBuffer1 = NULL;
    void* dBuffer2 = NULL;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
     
    CHECK_CUSPARSE(cusparseCreate(&handle))
    CHECK_CUSPARSE(cusparseCreateDnMat(&tmpA, num_A_rows, num_A_cols, lda, dA_pruned, type, order))
    CHECK_CUSPARSE(cusparseCreateCoo(&matA, num_A_rows, num_A_cols, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, type)) // Create sparse matrix A in Coo format   
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize1))
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1))
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer1)) // execute Dense to Sparse conversion                                                                                                                         
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp, &nnz))    // get number of non-zero elements
    CHECK_CUDA(cudaMalloc((void**)&d_coo_rows, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_coo_columns, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_coo_values, nnz * sizeof(T)))
    // reset row indices, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCooSetPointers(matA, d_coo_rows, d_coo_columns, d_coo_values))
    // execute Dense to Sparse conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer1))
            
    // -----------------------------------------------------------------------------------------
    // Create dense matrix 
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, num_B_rows, num_B_cols, ldb, dB, type, order))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, num_C_rows, num_C_cols, ldc, dC, type, order))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, opA, opB, &alpha, matA, matB, &beta, matC, compute_type, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize2))
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2))
    
    // perform matrix multiplication SpMM
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CHECK_CUSPARSE(cusparseSpMM(handle, opA, opB, &alpha, matA, matB, &beta, matC, compute_type, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer2))
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cusparseCOO spending time : " << ms << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    CHECK_CUDA(cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost))
     
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(tmpA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    // free memory
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(d_coo_rows))
    CHECK_CUDA(cudaFree(d_coo_columns))
    CHECK_CUDA(cudaFree(d_coo_values))
    CHECK_CUDA(cudaFree(dA_pruned))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))

    return EXIT_SUCCESS;
}

template <typename T>
int cusMatmulCsr(T* hA_pruned, T* hB, T* hC, const int m, const int n, const int k)
{
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type = CUDA_R_32F;
    auto          compute_type = CUDA_R_32F;
    bool     is_rowmajor = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows = (isA_transposed) ? k : m;
    auto     num_A_cols = (isA_transposed) ? m : k;
    auto     num_B_rows = (isB_transposed) ? n : k;
    auto     num_B_cols = (isB_transposed) ? k : n;
    auto     num_C_rows = m;
    auto     num_C_cols = n;
    auto     lda = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size = A_height * lda * sizeof(T);
    auto     B_size = B_height * ldb * sizeof(T);
    auto     C_size = C_height * ldc * sizeof(T);
      
    float alpha = 1.0f;
    float beta = 0.0f;

    // Device memory management
    T* dA_pruned, * dB, * dC;
    CHECK_CUDA(cudaMalloc((void**)&dA_pruned, A_size))
    CHECK_CUDA(cudaMalloc((void**)&dB, B_size))
    CHECK_CUDA(cudaMalloc((void**)&dC, C_size))
    CHECK_CUDA(cudaMemcpy(dA_pruned, hA_pruned, A_size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))

    int* d_csr_offsets, * d_csr_columns;
    T* d_csr_values;
    CHECK_CUDA(cudaMalloc((void**)&d_csr_offsets, (num_A_rows + 1) * sizeof(int)))

    // convert dense matrix tmpA --> sparse matrix matA in CSR format
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t tmpA, matB, matC;
    cusparseSpMatDescr_t matA;
    void* dBuffer1 = NULL;
    void* dBuffer2 = NULL;
    size_t               bufferSize1 = 0;
    size_t               bufferSize2 = 0;

    CHECK_CUSPARSE(cusparseCreate(&handle))
    CHECK_CUSPARSE(cusparseCreateDnMat(&tmpA, num_A_rows, num_A_cols, lda, dA_pruned, type, order))
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, num_A_rows, num_A_cols, 0, d_csr_offsets, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, type)) // Create sparse matrix A in Csr format   
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize1))
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1))
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer1)) // execute Dense to Sparse conversion                                                                                                                         
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp, &nnz))    // get number of non-zero elements   
    CHECK_CUDA(cudaMalloc((void**)&d_csr_columns, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_csr_values, nnz * sizeof(T)))
    // reset row indices, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(matA, d_csr_offsets, d_csr_columns, d_csr_values))
    // execute Dense to Sparse conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer1))
    // -----------------------------------------------------------------------------------------
    // Create dense matrix 
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, num_B_rows, num_B_cols, ldb, dB, type, order))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, num_C_rows, num_C_cols, ldc, dC, type, order))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, opA, opB, &alpha, matA, matB, &beta, matC, compute_type, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize2))
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2))

    // execute SpMM
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CHECK_CUSPARSE(cusparseSpMM(handle, opA, opB, &alpha, matA, matB, &beta, matC, compute_type, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer2))
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cusparseCSR spending time : " << ms << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost))

    
 /*   T* v = (T*)malloc(sizeof(T) * nnz);
    int* o = (int*)malloc(sizeof(int) * (num_A_rows + 1));
    int* c = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(v, d_csr_values, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(o, d_csr_offsets, sizeof(int) * (num_A_rows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_csr_columns, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    

    std::cout << "\nnnz -->\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;  std::cout << "\no -->";
    for (int i = 0; i < num_A_rows + 1; i++) {
        std::printf("%d ", o[i]);
    }
    std::cout << std::endl;  std::cout << "\nc -->";
    for (int i = 0; i < nnz; i++) {
        std::printf("%d ", c[i]);
    }
    
    free(v);
    free(o);
    free(c);*/


    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(tmpA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    // free memory
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(d_csr_offsets))
    CHECK_CUDA(cudaFree(d_csr_columns))
    CHECK_CUDA(cudaFree(d_csr_values))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))

    return EXIT_SUCCESS;
}

template <typename T>
int cusMatmulCsc(T* hA_pruned, T* hB, T* hC, const int m, const int n, const int k)
{
    auto          orderA = CUSPARSE_ORDER_COL;
    auto          orderB = CUSPARSE_ORDER_ROW;
    auto          orderC = CUSPARSE_ORDER_ROW;
    auto          opA = CUSPARSE_OPERATION_TRANSPOSE;
    auto          opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type = CUDA_R_32F;
    auto          compute_type = CUDA_R_32F;
    bool     is_rowmajorA = (orderA == CUSPARSE_ORDER_ROW);
    bool     is_rowmajorB = (orderB == CUSPARSE_ORDER_ROW);
    bool     is_rowmajorC = (orderC == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows = (isA_transposed) ? k : m;
    auto     num_A_cols = (isA_transposed) ? m : k;
    auto     num_B_rows = (isB_transposed) ? n : k;
    auto     num_B_cols = (isB_transposed) ? k : n;
    auto     num_C_rows = m;
    auto     num_C_cols = n;
    auto     lda = (is_rowmajorA) ? num_A_cols : num_A_rows;
    auto     ldb = (is_rowmajorB) ? num_B_cols : num_B_rows;
    auto     ldc = (is_rowmajorC) ? num_C_cols : num_C_rows;
    auto     A_height = (is_rowmajorA) ? num_A_rows : num_A_cols;
    auto     B_height = (is_rowmajorB) ? num_B_rows : num_B_cols;
    auto     C_height = (is_rowmajorC) ? num_C_rows : num_C_cols;
    auto     A_size = A_height * lda * sizeof(T);
    auto     B_size = B_height * ldb * sizeof(T);
    auto     C_size = C_height * ldc * sizeof(T);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Device memory management
    T* dA_pruned, * dB, * dC;
    CHECK_CUDA(cudaMalloc((void**)&dA_pruned, A_size))
    CHECK_CUDA(cudaMalloc((void**)&dB, B_size))
    CHECK_CUDA(cudaMalloc((void**)&dC, C_size))
    CHECK_CUDA(cudaMemcpy(dA_pruned, hA_pruned, A_size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))

    int* d_csc_offsets, * d_csc_rows;
    T* d_csc_values;
    CHECK_CUDA(cudaMalloc((void**)&d_csc_offsets, (num_A_cols + 1) * sizeof(int)))

    // convert dense matrix tmpA --> sparse matrix matA in CSC format
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t tmpA, matB, matC;
    cusparseSpMatDescr_t matA;
    void* dBuffer1 = NULL;
    void* dBuffer2 = NULL;
    size_t               bufferSize1 = 0;
    size_t               bufferSize2 = 0;

    CHECK_CUSPARSE(cusparseCreate(&handle))
    CHECK_CUSPARSE(cusparseCreateDnMat(&tmpA, num_A_rows, num_A_cols, lda, dA_pruned, type, orderA))
    CHECK_CUSPARSE(cusparseCreateCsc(&matA, num_A_rows, num_A_cols, 0, d_csc_offsets, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, type)) // Create sparse matrix A in Coo format   
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize1))
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1))
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer1)) // execute Dense to Sparse conversion                                                                                                                         
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp, &nnz))    // get number of non-zero elements  
    CHECK_CUDA(cudaMalloc((void**)&d_csc_rows, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_csc_values, nnz * sizeof(T)))
    // reset row indices, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCscSetPointers(matA, d_csc_offsets, d_csc_rows, d_csc_values))
    // execute Dense to Sparse conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, tmpA, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer1))
    // -----------------------------------------------------------------------------------------
    // Create dense matrix 
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, num_B_rows, num_B_cols, ldb, dB, type, orderB))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, num_C_rows, num_C_cols, ldc, dC, type, orderC))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, opA, opB, &alpha, matA, matB, &beta, matC, compute_type, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize2))
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2))

    // execute SpMM
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CHECK_CUSPARSE(cusparseSpMM(handle, opA, opB, &alpha, matA, matB, &beta, matC, compute_type, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer2))
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cusparseCSC spending time : " << ms << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost))

    
    //T* v = (T*)malloc(sizeof(T) * nnz);
    //int* o = (int*)malloc(sizeof(int) * (num_A_cols + 1));
    //int* r = (int*)malloc(sizeof(int) * nnz);
    //cudaMemcpy(v, d_csc_values, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    //cudaMemcpy(o, d_csc_offsets, sizeof(int) * (num_A_cols + 1), cudaMemcpyDeviceToHost);
    //cudaMemcpy(r, d_csc_rows, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    //

    //std::cout << "\nnnz -->\n";
    //for (int i = 0; i < nnz; i++) {
    //    std::cout << v[i] << " ";
    //}
    //std::cout << std::endl; std::cout << "\no -->\n";
    //for (int i = 0; i < num_A_cols + 1; i++) {
    //    std::printf("%d ", o[i]);
    //}
    //std::cout << std::endl; std::cout << "\nr -->\n";
    //for (int i = 0; i < nnz; i++) {
    //    std::printf("%d ",r[i]);
    //}

    //free(v);
    //free(o);
    //free(r);
    

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(tmpA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    // free memory
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(d_csc_offsets))
    CHECK_CUDA(cudaFree(d_csc_rows))
    CHECK_CUDA(cudaFree(d_csc_values))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))

    return EXIT_SUCCESS;
}