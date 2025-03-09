//-- -- -- -- -- -- -- -- --Double_Buffer_SMem &RMem_gemm-- -- -- -- -- -- -- -- -- //
#pragma once
template <unsigned int M_per_BLOCK,  // 128
          unsigned int K_per_BLOCK,  // 8
          unsigned int N_per_BLOCK,  // 128
          unsigned int Y_per_THREAD, // width of block of C that each thread calculate
          unsigned int X_per_THREAD> // height of block of C that each thread calculate

__global__ void
gemm_v8(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int X_Thread_Num_per_Block = N_per_BLOCK / X_per_THREAD;
    // const int Y_Thread_Num_per_Block = M_per_BLOCK / Y_per_THREAD;

    // per thread calc
    float sum[Y_per_THREAD][X_per_THREAD] = {0.0f};
    float a_reg[2][Y_per_THREAD] = {0.0f};
    float b_reg[2][X_per_THREAD] = {0.0f};

    // SMem
    __shared__ float A_block[2][K_per_BLOCK][M_per_BLOCK];
    __shared__ float B_block[2][K_per_BLOCK][N_per_BLOCK];

    // d_C ptr
    int row = by * M_per_BLOCK;
    int col = bx * N_per_BLOCK;

    // tid in thread block
    const int tid = ty * X_Thread_Num_per_Block + tx;

    const int A_tile_per_row_thread = K_per_BLOCK / 4; // 2
    const int B_tile_per_row_thread = N_per_BLOCK / 4; // 64

    // matrix block : tile(ty,tx)
    int A_tile_ty = tid / A_tile_per_row_thread;
    int A_tile_tx = tid % A_tile_per_row_thread;

    int B_tile_ty = tid / B_tile_per_row_thread;
    int B_tile_tx = tid % B_tile_per_row_thread;

    // tmp for loading GMem2SMem
    float a_tmp[4];
    // float b_tmp[4];

    // reg�����ܲ�Ҫbank conflict
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4;
    // warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
    const int b_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;

    // �ڽ���ѭ��֮ǰ �ȴ� GMem Ԥȡ��һ���ֵ� SMem �� Ҳ����ԭ�ȵ�ѭ���ĵ�һ���ֵĲ���
    // ֮�����ѭ�� �� GMem ��ȡ�����ݴ��� SMem ��ʣ�²��֣��������һ���ֵ�sum ��ͬ�� Ҳ���ǰ�ԭ�ȵ�����ͬ����ζһ��ͬ��
    int write = 0;
    int read = 0;

    // GMem2SMem
    FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, A_tile_tx * 4, K)]);
    A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = a_tmp[0];
    A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = a_tmp[1];
    A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = a_tmp[2];
    A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = a_tmp[3];

    FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
        FETCH_FLOAT4(d_B[OFFSET(B_tile_ty, col + B_tile_tx * 4, N)]);

    __syncthreads();
    write ^= 1;

    //  SMem2RMem
    FETCH_FLOAT4(a_reg[0][0]) = FETCH_FLOAT4(A_block[read][0][a_tile_index]);
    FETCH_FLOAT4(a_reg[0][4]) = FETCH_FLOAT4(A_block[read][0][a_tile_index + 64]);
    FETCH_FLOAT4(b_reg[0][0]) = FETCH_FLOAT4(B_block[read][0][b_tile_index]);
    FETCH_FLOAT4(b_reg[0][4]) = FETCH_FLOAT4(B_block[read][0][b_tile_index + 64]);

    int tile_idx = 0;
    // �����ѭ��
    do
    {
        tile_idx += K_per_BLOCK;

        for (int inner_k = 0; inner_k < K_per_BLOCK - 1; ++inner_k)
        {
            FETCH_FLOAT4(a_reg[(inner_k + 1) % 2][0]) = FETCH_FLOAT4(A_block[read][inner_k + 1][a_tile_index]);
            FETCH_FLOAT4(a_reg[(inner_k + 1) % 2][4]) = FETCH_FLOAT4(A_block[read][inner_k + 1][a_tile_index + 64]);
            FETCH_FLOAT4(b_reg[(inner_k + 1) % 2][0]) = FETCH_FLOAT4(B_block[read][inner_k + 1][b_tile_index]);
            FETCH_FLOAT4(b_reg[(inner_k + 1) % 2][4]) = FETCH_FLOAT4(B_block[read][inner_k + 1][b_tile_index + 64]);
            // ����Ԥȡ�ĺ�
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[i][j] += a_reg[inner_k % 2][i] * b_reg[inner_k % 2][j];
                }
            }
        }
        read ^= 1;
        if (tile_idx < K)
        {
            FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, tile_idx + A_tile_tx * 4, K)]);
            A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = a_tmp[0];
            A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = a_tmp[1];
            A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = a_tmp[2];
            A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = a_tmp[3];

            FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
                FETCH_FLOAT4(d_B[OFFSET(B_tile_ty + tile_idx, col + B_tile_tx * 4, N)]);
            write ^= 1;
        }

        __syncthreads();
        FETCH_FLOAT4(a_reg[0][0]) = FETCH_FLOAT4(A_block[read][0][a_tile_index]);
        FETCH_FLOAT4(a_reg[0][4]) = FETCH_FLOAT4(A_block[read][0][a_tile_index + 64]);
        FETCH_FLOAT4(b_reg[0][0]) = FETCH_FLOAT4(B_block[read][0][b_tile_index]);
        FETCH_FLOAT4(b_reg[0][4]) = FETCH_FLOAT4(B_block[read][0][b_tile_index + 64]);
        for (int i = 0; i < Y_per_THREAD; ++i)
        {
            for (int j = 0; j < X_per_THREAD; ++j)
            {
                sum[i][j] += a_reg[1][i] * b_reg[1][j];
            }
        }

    } while (tile_idx < K);

    // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������

    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i, col + b_tile_index, N)]) = FETCH_FLOAT4(sum[i][0]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i, col + b_tile_index + 64, N)]) = FETCH_FLOAT4(sum[i][4]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i + 64, col + b_tile_index, N)]) = FETCH_FLOAT4(sum[i + 4][0]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i + 64, col + b_tile_index + 64, N)]) = FETCH_FLOAT4(sum[i + 4][4]);
    }
}


// template <
//     const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
//     const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
//     const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
//     const int THREAD_SIZE_Y, // height of block of C that each thread calculate
//     const int THREAD_SIZE_X  // width of block of C that each thread calculate
//     >
    
// __global__ void
// gemm_v8(
//     int M, int K, int N,
//     float *__restrict__ A,
//     float *__restrict__ B,
//     float *__restrict__ C){
//     // thread Block index
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     // Thread index
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
    
//     // the threads number in Block of X,Y
//     const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;// x�����ϵ��߳���
//     const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;// y�����ϵ��߳���
//     const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;// 16 * 16 ���߳�

//     // thread id in cur Block
//     const int tid = ty * THREAD_X_PER_BLOCK + tx;

//     // shared memory
//     __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
//     __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
//     // registers for C
//     float accum[THREAD_SIZE_Y][THREAD_SIZE_X];
//     #pragma unroll//????????????
//     for(int i=0; i<THREAD_SIZE_Y; i++){
//         #pragma unroll
//         for(int j=0; j<THREAD_SIZE_X; j++){
//             accum[i][j]=0.0;
//         }
//     }
//     // registers for A and B
//     float frag_a[2][THREAD_SIZE_Y];
//     float frag_b[2][THREAD_SIZE_X];


//     // registers load global memory ??????????? 
//     const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4); // 1
//     const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4); // 1 
//     float ldg_a_reg[4*ldg_num_a];
//     float ldg_b_reg[4*ldg_num_b];

//     // threads number in one row
//     const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
//     const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

//     // row number and col number that needs to be loaded by this thread
//     const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
//     const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

//     const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
//     const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

//     // row stride that thread uses to load multiple rows of a tile
//     const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
//     const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

//     A = &A[(BLOCK_SIZE_M * by)* K];
//     B = &B[BLOCK_SIZE_N * bx];

//     //load index of the tile
//     const int warp_id = tid / 32;
//     const int lane_id = tid % 32;
//     const int a_tile_index =  warp_id/2*16 + lane_id/8*4; 
//     //warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
//     const int b_tile_index =  warp_id%2*32 + lane_id%8*4; 
//     //(lane_id % 16) * 4; // (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;
    
//     // transfer first tile from global mem to shared mem
//     // load A from global memory to shared memory
//     #pragma unroll
//     for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
//         int ldg_index = i / A_TILE_ROW_STRIDE * 4;
//         FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
//             A_TILE_ROW_START + i, // row
//             A_TILE_COL, // col
//             K )]);
//         As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
//         As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
//         As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
//         As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
//     }
//     // load B from global memory to shared memory
//     #pragma unroll
//     for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
//         FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
//                 B_TILE_ROW_START + i, // row
//                 B_TILE_COL, // col
//                 N )]);
//     }
//     __syncthreads();
    
//     // load A from shared memory to register
//     FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[0][0][a_tile_index]);
//     FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[0][0][a_tile_index + 64]);
    
//     // load B from shared memory to register
//     FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[0][0][b_tile_index]);
//     FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[0][0][b_tile_index + 64]);
    
//     int write_stage_idx = 1;
//     int tile_idx = 0;








//     do{
//         // next tile index
//         tile_idx += BLOCK_SIZE_K;
//         // load next tile from global mem
//         if(tile_idx< K){
//             #pragma unroll
//             for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
//                 int ldg_index = i / A_TILE_ROW_STRIDE * 4;
//                 FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
//                     A_TILE_ROW_START + i, // row
//                     A_TILE_COL + tile_idx, // col
//                     K )]);
//             }
//             #pragma unroll
//             for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
//                 int ldg_index = i / B_TILE_ROW_STRIDE * 4;
//                 FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
//                     tile_idx + B_TILE_ROW_START + i, // row
//                     B_TILE_COL, // col
//                     N )]);
//             }
//         }

//         int load_stage_idx = write_stage_idx ^ 1;




        
//         #pragma unroll
//         for(int j=0; j<BLOCK_SIZE_K - 1; ++j){
//             // load next tile from shared mem to register 
//             // load A from shared memory to register
//             FETCH_FLOAT4(frag_a[(j+1)%2][0]) = FETCH_FLOAT4(As[load_stage_idx][(j+1)][a_tile_index]);
//             FETCH_FLOAT4(frag_a[(j+1)%2][4]) = FETCH_FLOAT4(As[load_stage_idx][(j+1)][a_tile_index + 64]);
//             // load B from shared memory to register
//             FETCH_FLOAT4(frag_b[(j+1)%2][0]) = FETCH_FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index]);
//             FETCH_FLOAT4(frag_b[(j+1)%2][4]) = FETCH_FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index + 64]);
//             // compute C THREAD_SIZE_X x THREAD_SIZE_Y
//             #pragma unroll
//             for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
//                 #pragma unroll
//                 for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
//                     accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
//                 }
//             }
//         }




//         if(tile_idx < K){
//             // load A from global memory to shared memory
//             #pragma unroll
//             for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
//                 int ldg_index = i / A_TILE_ROW_STRIDE * 4;
//                 As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
//                 As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
//                 As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
//                 As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
//             }
//             // load B from global memory to shared memory
//             #pragma unroll
//             for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
//                 int ldg_index = i / B_TILE_ROW_STRIDE * 4;
//                 FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
//             }
//             // use double buffer, only need one sync
//             __syncthreads();
//             // switch
//             write_stage_idx ^= 1;
//         }




//         // load first tile from shared mem to register of next iter
//         // load A from shared memory to register
//         FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[load_stage_idx^1][0][a_tile_index]);
//         FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[load_stage_idx^1][0][a_tile_index + 64]);
//         // load B from shared memory to register
//         FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][b_tile_index]);
//         FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][b_tile_index + 64]);
//         // compute C THREAD_SIZE_X x THREAD_SIZE_Y
//         #pragma unroll
//         for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
//             #pragma unroll
//             for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
//                 accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
//             }
//         }
//     }while(tile_idx< K);
    















//     const int c_block_row = a_tile_index;
//     const int c_block_col = b_tile_index;

//     //store C00 block
//     for(int i=0; i<4; i++){
//       FETCH_FLOAT4(C[OFFSET(
//         BLOCK_SIZE_M * by + c_block_row + i,
//         BLOCK_SIZE_N * bx + c_block_col,
//         N)]) = FETCH_FLOAT4(accum[i][0]);
//     }
//     //store C01 block
//     for(int i=0; i<4; i++){
//       FETCH_FLOAT4(C[OFFSET(
//         BLOCK_SIZE_M * by + c_block_row + i,
//         BLOCK_SIZE_N * bx + c_block_col + 64,
//         N)]) = FETCH_FLOAT4(accum[i][4]);
//     }
//     //store C10 block
//     for(int i=0; i<4; i++){
//       FETCH_FLOAT4(C[OFFSET(
//         BLOCK_SIZE_M * by + c_block_row + 64 + i,
//         BLOCK_SIZE_N * bx + c_block_col,
//         N)]) = FETCH_FLOAT4(accum[i+4][0]);
//     }
//     //store C11 block
//     for(int i=0; i<4; i++){
//       FETCH_FLOAT4(C[OFFSET(
//         BLOCK_SIZE_M * by + c_block_row + 64 + i,
//         BLOCK_SIZE_N * bx + c_block_col + 64,
//         N)]) = FETCH_FLOAT4(accum[i+4][4]);
//     }
// }
