add_executable(test_transpose 
  test_transpose.cu 
  utlis.cu)
target_link_libraries(test_transpose PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(test_transpose PRIVATE $<$<CONFIG:Debug>:-G>)
endif()

# utlis.cu is a separate file containing helper functions used by test_transpose.cu.