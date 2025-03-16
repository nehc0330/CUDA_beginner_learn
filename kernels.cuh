#pragma once

#include"gemm_v1.cuh"

#include"gemm_v2.cuh"

// 共享存储的内存访问模式可能不是最优的，导致平均每1.3次发生一次存储体冲突，总共在16777216次共享存储请求中发生了5529808次存储体冲突。
// 这代表了共享存储请求中整体22307024个波前的24.79%。

// 理论占用率
// 预估加速比: 12.21%
// 根据其占用率，该内核每调度器可以发出的理论 warp 数量为 8.00，低于硬件最大值 12。
// 该内核的理论占用率 (66.7%) 受所需寄存器数量的限制。该内核的理论占用率 (66.7%) 受所需共享内存数量的限制。
// 该内核的理论占用率 (66.7%) 受每个块中 warp 数量的限制。

#include"gemm_v3.cuh"
#include"gemm_v4.cuh"
#include"gemm_v5.cuh"
#include"gemm_v6.cuh"
#include"gemm_v7.cuh"
#include"gemm_v8.cuh"