add_executable(test_gemm 
    test_gemm.cu 
    utils.cu  # ������һ��
)
   
target_link_libraries(test_gemm PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(test_gemm PRIVATE $<$<CONFIG:Debug>:-G>)
endif()