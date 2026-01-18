#include <cooperative_groups.h>

namespace ppln::collision {

    namespace cg = cooperative_groups;





    #define XARM7_APPROX_SPHERE_COUNT 15
    #define XARM7_APPROX_JOINT_COUNT 8
    #define XARM7_APPROX_SELF_CC_RANGE_COUNT 8
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    #define BATCH_SIZE G_BATCH_SIZE
    
    __device__ __constant__ float4 xarm7_approx_spheres_array[15] = {
        { -0.005656f, 0.00253f, 0.074197f, 0.105153f },
        { 0.00086f, 0.015176f, -0.032332f, 0.10491f },
        { 0.001527f, -0.078131f, 0.042732f, 0.144777f },
        { 0.027181f, -0.008649f, -0.029312f, 0.104016f },
        { 0.043174f, -0.066506f, 0.031464f, 0.141191f },
        { -0.002556f, 0.017912f, -0.072019f, 0.118989f },
        { 0.041217f, 0.018877f, 0.013476f, 0.092715f },
        { 0.000149f, -0.006938f, -0.013358f, 0.046742f },
        { -0.002264f, 0.000217f, 0.054296f, 0.067689f },
        { 0.000567f, 0.045888f, 0.075449f, 0.041974f },
        { -0.000714f, 0.059004f, 0.1293f, 0.038679f },
        { -0.000346f, 0.038164f, 0.094632f, 0.036745f },
        { 0.00024f, -0.037075f, 0.095476f, 0.037078f },
        { 0.000215f, -0.045222f, 0.076468f, 0.04216f },
        { -0.000448f, -0.0602f, 0.130247f, 0.037849f }
    };
    
    __device__ __constant__ float xarm7_approx_fixed_transforms[] = {
        // joint 0
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.267,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, -0.0, 0.0, 0.0,
        0.0, -4e-06, 1.0, 0.0,
        0.0, -1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.0,
        0.0, -4e-06, -1.0, -0.293,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.0525,
        0.0, -4e-06, -1.0, 0.0,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, 0.0775,
        0.0, -4e-06, -1.0, -0.3425,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.0,
        0.0, -4e-06, -1.0, 0.0,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, -0.0, 0.0, 0.076,
        0.0, -4e-06, 1.0, 0.097,
        0.0, -1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };
    
    __device__ __constant__ int xarm7_approx_sphere_to_joint[15] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    };
    
    __device__ __constant__ int xarm7_approx_joint_to_sphere_count[8] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        8
    };
    
    __device__ __constant__ int xarm7_approx_flattened_joint_to_spheres[15] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14
    };
    
    __device__ __constant__ int xarm7_approx_joint_types[] = {
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        5
    };
    
    __device__ __constant__ int xarm7_approx_self_cc_ranges[8][3] = {
        { 0, 4, 14 },
        { 1, 5, 14 },
        { 2, 4, 14 },
        { 3, 10, 10 },
        { 3, 14, 14 },
        { 4, 8, 14 },
        { 5, 8, 9 },
        { 5, 13, 13 }
    };
    
    __device__ __constant__ int xarm7_approx_joint_parents[8] = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    __device__ __constant__ int xarm7_approx_T_memory_idx[8] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    
    __device__ __constant__ int xarm7_approx_dfs_order[8] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    __device__ __constant__ int xarm7_approx_joint_id_to_dof[8] = {
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    template <>
    __device__ void fk_approx<ppln::robots::Xarm7>(
        const float* q,
        float* sphere_pos_approx, // 15 spheres x 16 robots x 3 coordinates (each column is a robot)
        float *T, // 16 robots x 1 x 4x4 transform matrix , column major
        const int tid
    )
    {
        auto cta = cg::this_thread_block();
        auto tile = cg::tiled_partition<4>(cta);
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid % 4;
        const int batch_ind = tid / 4;
    
        int T_offset = batch_ind * 1 * 16;
        float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
        float *T_base = T + T_offset; // 4x4 transform matrix for the batch
        
        #pragma unroll
        for (int i = 0; i < 1; ++i) {
            float *T_col_i = T_base + i * 16 + col_ind * 4;
            for (int r=0; r<4; r++) {
                T_col_i[r] = 0.0f;
            }
            T_col_i[col_ind] = 1.0f;
        }
        tile.sync();
    
        int transformed_sphere_ind = 0;
    
        for (int j = 0; j < XARM7_APPROX_JOINT_COUNT; ++j) {
            int i = xarm7_approx_dfs_order[j];
            float T_col_tmp[4];
            int parent_idx = xarm7_approx_joint_parents[i];
            int T_memory_idx_parent = xarm7_approx_T_memory_idx[parent_idx];
            int T_memory_idx = xarm7_approx_T_memory_idx[i];
            int q_idx = xarm7_approx_joint_id_to_dof[i];
            if (j > 0) {
                int ft_addr_start = i * 16;
                int joint_type = xarm7_approx_joint_types[i];
    
                if (joint_type <= Z_PRISM) {
                    prism_fn(&xarm7_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
                }
                else if (joint_type == X_ROT) {
                    xrot_fn(&xarm7_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Y_ROT) {
                    yrot_fn(&xarm7_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Z_ROT) {
                    zrot_fn(&xarm7_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                
                for (int r=0; r<4; r++){
                    T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
                }
                for (int r=0; r<4; r++){
                    T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
                }
            }
            tile.sync();
            int sphere_count = xarm7_approx_joint_to_sphere_count[i];
            for (int s = transformed_sphere_ind + col_ind; s < transformed_sphere_ind + sphere_count; s += 4) {
                int sphere_ind = xarm7_approx_flattened_joint_to_spheres[s];
                for (int c = 0; c < 3; c++) {
                    sphere_pos_approx[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + c] = 
                        T_base[T_memory_idx*16 + c] * xarm7_approx_spheres_array[sphere_ind].x +
                        T_base[T_memory_idx*16 + c + M] * xarm7_approx_spheres_array[sphere_ind].y +
                        T_base[T_memory_idx*16 + c + M*2] * xarm7_approx_spheres_array[sphere_ind].z +
                        T_base[T_memory_idx*16 + c + M*3];
                }
            }
            transformed_sphere_ind += sphere_count;
            tile.sync();
        }
    }
    
    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check_approx<ppln::robots::Xarm7>(float* sphere_pos_approx, int* joint_in_collision, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;
        for (int i = thread_ind; i < XARM7_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
            int sphere_1_ind = xarm7_approx_self_cc_ranges[i][0];
            float sphere_1[3] = {
                sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            for (int j = xarm7_approx_self_cc_ranges[i][1]; j <= xarm7_approx_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], xarm7_approx_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], xarm7_approx_spheres_array[j].w
                )){
                    atomicOr((int*)&joint_in_collision[20*batch_ind + xarm7_approx_sphere_to_joint[sphere_1_ind]], 2);
                    out = false;
                }
            } 
        }
        return out;
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Xarm7>(float* sphere_pos_approx, int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;
    
        for (int i = thread_ind; i < XARM7_APPROX_SPHERE_COUNT; i += 4){
            // sphere i, robot batch_ind (32 robots)
            if (i > 0 &&
                sphere_environment_in_collision(
                    env,
                    sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                    xarm7_approx_spheres_array[i].w
                )
            ) {
                atomicOr((int*)&joint_in_collision[20*batch_ind + xarm7_approx_sphere_to_joint[i]], 1);
                out = false;
            } 
        }
        return out;
    }
    
    
    
    
    #define XARM7_SPHERE_COUNT 74
    #define XARM7_JOINT_COUNT 8
    #define XARM7_SELF_CC_RANGE_COUNT 29
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    #define BATCH_SIZE G_BATCH_SIZE
    
    __device__ __constant__ float4 xarm7_spheres_array[74] = {
        { -0.009247f, -0.000596f, 0.046198f, 0.090156f },
        { -6.3e-05f, -0.000199f, 0.108338f, 0.073725f },
        { -0.000264f, 0.016582f, -0.000398f, 0.081809f },
        { 5.5e-05f, -0.003638f, -0.06871f, 0.07187f },
        { -0.000318f, -0.13074f, 0.001376f, 0.08038f },
        { 0.003099f, -0.094223f, 0.075337f, 0.078078f },
        { 0.007966f, -0.001553f, 0.09806f, 0.060525f },
        { -0.015469f, -0.002019f, 0.096291f, 0.061347f },
        { 0.004163f, 0.014404f, -0.052724f, 0.078294f },
        { 0.050608f, -0.001919f, -0.000687f, 0.084246f },
        { 0.07675f, -0.13282f, -0.00102f, 0.061902f },
        { 0.07131f, -0.073023f, 0.065105f, 0.065406f },
        { 0.078721f, -0.087475f, 0.009699f, 0.065197f },
        { 0.026035f, -0.032353f, 0.083618f, 0.056541f },
        { 0.00122f, 0.002187f, 0.086713f, 0.054491f },
        { -0.000183f, 0.045752f, -0.0053f, 0.059255f },
        { -6e-06f, 0.0f, -0.145661f, 0.050339f },
        { -0.000972f, 0.017486f, -0.140641f, 0.049716f },
        { 0.000121f, 0.04511f, -0.111783f, 0.042043f },
        { 0.009606f, 0.06927f, -0.070865f, 0.038992f },
        { -0.013871f, 0.069029f, -0.071609f, 0.038827f },
        { 0.074781f, 0.019259f, 0.000652f, 0.068595f },
        { 0.008324f, 0.005169f, 0.024117f, 0.059027f },
        { 2e-06f, 6e-06f, -0.0135f, 0.040075f },
        { -0.00285f, -0.014902f, -0.013501f, 0.039071f },
        { -0.001932f, -4.9e-05f, 0.05453f, 0.067392f },
        { 0.0f, 0.069546f, 0.076739f, 0.010695f },
        { 0.0f, 0.078215f, 0.083552f, 0.010423f },
        { 0.0f, 0.056504f, 0.070388f, 0.011066f },
        { 0.0f, 0.070567f, 0.100939f, 0.007677f },
        { 0.0f, 0.074642f, 0.093332f, 0.009907f },
        { 0.0f, 0.032523f, 0.058588f, 0.011425f },
        { 0.0f, 0.018722f, 0.05711f, 0.01042f },
        { 0.0f, 0.044807f, 0.064173f, 0.010247f },
        { 0.00774f, 0.049417f, 0.151326f, 0.014111f },
        { -0.006215f, 0.049568f, 0.152732f, 0.014052f },
        { 0.004243f, 0.051288f, 0.133434f, 0.015265f },
        { 0.00113f, 0.052746f, 0.123351f, 0.017574f },
        { -0.008289f, 0.050252f, 0.133854f, 0.014876f },
        { 0.000368f, 0.061255f, 0.110026f, 0.011919f },
        { 0.004436f, 0.069585f, 0.102318f, 0.009289f },
        { -0.004329f, 0.069284f, 0.10258f, 0.009733f },
        { -0.009387f, 0.051991f, 0.112161f, 0.012727f },
        { -0.008075f, 0.04043f, 0.097481f, 0.013041f },
        { 0.008877f, 0.052479f, 0.11253f, 0.012112f },
        { 0.006817f, 0.040426f, 0.098626f, 0.0133f },
        { -0.011815f, 0.027891f, 0.083731f, 0.010852f },
        { 0.008633f, 0.02813f, 0.083743f, 0.012241f },
        { 0.011573f, 0.019784f, 0.074822f, 0.007637f },
        { -0.011682f, 0.020801f, 0.074534f, 0.007466f },
        { 0.011241f, -0.03464f, 0.089045f, 0.011703f },
        { -0.009162f, -0.052209f, 0.11202f, 0.012715f },
        { -0.009056f, -0.030422f, 0.086615f, 0.012718f },
        { 0.008495f, -0.023264f, 0.077108f, 0.012422f },
        { -0.012246f, -0.021529f, 0.075221f, 0.009182f },
        { -0.008764f, -0.040782f, 0.09932f, 0.011657f },
        { 0.008471f, -0.043119f, 0.100054f, 0.012768f },
        { 0.010196f, -0.053459f, 0.113503f, 0.010934f },
        { 0.0f, -0.068091f, 0.07698f, 0.01123f },
        { 0.0f, -0.072217f, 0.097749f, 0.010764f },
        { 0.0f, -0.077353f, 0.084376f, 0.011637f },
        { 0.0f, -0.02891f, 0.058567f, 0.009278f },
        { 0.0f, -0.017677f, 0.056703f, 0.009508f },
        { 0.0f, -0.037224f, 0.059301f, 0.009497f },
        { 0.0f, -0.054141f, 0.068885f, 0.012465f },
        { 0.0f, -0.041771f, 0.062595f, 0.009353f },
        { -0.003486f, -0.066996f, 0.104606f, 0.011436f },
        { 0.002997f, -0.067039f, 0.104544f, 0.011847f },
        { -0.000327f, -0.055393f, 0.116692f, 0.01371f },
        { -0.00709f, -0.051379f, 0.132032f, 0.013301f },
        { 0.008732f, -0.050444f, 0.132635f, 0.013163f },
        { 0.003747f, -0.052715f, 0.14795f, 0.014362f },
        { -0.010799f, -0.049127f, 0.151679f, 0.012888f },
        { 0.004963f, -0.049733f, 0.157083f, 0.012513f }
    };
    
    __device__ __constant__ float xarm7_fixed_transforms[] = {
        // joint 0
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.267,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, -0.0, 0.0, 0.0,
        0.0, -4e-06, 1.0, 0.0,
        0.0, -1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.0,
        0.0, -4e-06, -1.0, -0.293,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.0525,
        0.0, -4e-06, -1.0, 0.0,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, 0.0775,
        0.0, -4e-06, -1.0, -0.3425,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.0,
        0.0, -4e-06, -1.0, 0.0,
        0.0, 1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, -0.0, 0.0, 0.076,
        0.0, -4e-06, 1.0, 0.097,
        0.0, -1.0, -4e-06, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };
    
    __device__ __constant__ int xarm7_sphere_to_joint[74] = {
        0,
        0,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    };
    
    __device__ __constant__ int xarm7_joint_to_sphere_count[8] = {
        2,
        2,
        4,
        2,
        5,
        6,
        2,
        51
    };
    
    __device__ __constant__ int xarm7_flattened_joint_to_spheres[74] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73
    };
    
    __device__ __constant__ int xarm7_joint_types[] = {
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        5
    };
    
    __device__ __constant__ int xarm7_self_cc_ranges[29][3] = {
        { 0, 10, 73 },
        { 1, 10, 73 },
        { 2, 15, 73 },
        { 3, 15, 73 },
        { 4, 10, 73 },
        { 5, 10, 73 },
        { 6, 10, 73 },
        { 7, 10, 73 },
        { 8, 34, 41 },
        { 8, 66, 73 },
        { 9, 34, 41 },
        { 9, 66, 73 },
        { 10, 25, 73 },
        { 11, 25, 73 },
        { 12, 25, 73 },
        { 13, 25, 73 },
        { 14, 25, 73 },
        { 15, 25, 33 },
        { 15, 58, 65 },
        { 16, 25, 33 },
        { 16, 58, 65 },
        { 17, 25, 33 },
        { 17, 58, 65 },
        { 18, 25, 33 },
        { 18, 58, 65 },
        { 19, 25, 33 },
        { 19, 58, 65 },
        { 20, 25, 33 },
        { 20, 58, 65 }
    };
    
    __device__ __constant__ int xarm7_joint_parents[8] = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    __device__ __constant__ int xarm7_T_memory_idx[8] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    
    __device__ __constant__ int xarm7_dfs_order[8] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    __device__ __constant__ int xarm7_joint_id_to_dof[8] = {
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    template <>
    __device__ void fk<ppln::robots::Xarm7>(
        const float* q,
        float* sphere_pos, // 74 spheres x 16 robots x 3 coordinates (each column is a robot)
        float *T, // 16 robots x 1 x 4x4 transform matrix , column major
        const int tid
    )
    {
        auto cta = cg::this_thread_block();
        auto tile = cg::tiled_partition<4>(cta);
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid % 4;
        const int batch_ind = tid / 4;
    
        int T_offset = batch_ind * 1 * 16;
        float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
        float *T_base = T + T_offset; // 4x4 transform matrix for the batch
        
        #pragma unroll
        for (int i = 0; i < 1; ++i) {
            float *T_col_i = T_base + i * 16 + col_ind * 4;
            for (int r=0; r<4; r++) {
                T_col_i[r] = 0.0f;
            }
            T_col_i[col_ind] = 1.0f;
        }
        tile.sync();
    
        int transformed_sphere_ind = 0;
    
        for (int j = 0; j < XARM7_JOINT_COUNT; ++j) {
            int i = xarm7_dfs_order[j];
            float T_col_tmp[4];
            int parent_idx = xarm7_joint_parents[i];
            int T_memory_idx_parent = xarm7_T_memory_idx[parent_idx];
            int T_memory_idx = xarm7_T_memory_idx[i];
            int q_idx = xarm7_joint_id_to_dof[i];
            if (j > 0) {
                int ft_addr_start = i * 16;
                int joint_type = xarm7_joint_types[i];
    
                if (joint_type <= Z_PRISM) {
                    prism_fn(&xarm7_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
                }
                else if (joint_type == X_ROT) {
                    xrot_fn(&xarm7_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Y_ROT) {
                    yrot_fn(&xarm7_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Z_ROT) {
                    zrot_fn(&xarm7_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                
                for (int r=0; r<4; r++){
                    T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
                }
                for (int r=0; r<4; r++){
                    T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
                }
            }
            tile.sync();
            int sphere_count = xarm7_joint_to_sphere_count[i];
            for (int s = transformed_sphere_ind + col_ind; s < transformed_sphere_ind + sphere_count; s += 4) {
                int sphere_ind = xarm7_flattened_joint_to_spheres[s];
                for (int c = 0; c < 3; c++) {
                    sphere_pos[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + c] = 
                        T_base[T_memory_idx*16 + c] * xarm7_spheres_array[sphere_ind].x +
                        T_base[T_memory_idx*16 + c + M] * xarm7_spheres_array[sphere_ind].y +
                        T_base[T_memory_idx*16 + c + M*2] * xarm7_spheres_array[sphere_ind].z +
                        T_base[T_memory_idx*16 + c + M*3];
                }
            }
            transformed_sphere_ind += sphere_count;
            tile.sync();
        }
    }
    
    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check<ppln::robots::Xarm7>(float* sphere_pos, int* joint_in_collision, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool has_collision = false;
    
        for (int i = thread_ind; i < XARM7_SELF_CC_RANGE_COUNT; i += 4) {
            if (warp_any_active_mask(has_collision)) return false;
            int sphere_1_ind = xarm7_self_cc_ranges[i][0];
            if (!(joint_in_collision[20*batch_ind + xarm7_sphere_to_joint[sphere_1_ind]] & 2)) continue;
            float sphere_1[3] = {
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            for (int j = xarm7_self_cc_ranges[i][1]; j <= xarm7_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], xarm7_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], xarm7_spheres_array[j].w
                )){
                    //return false;
                    has_collision=true;
                }
            }
        }
        return !has_collision;
    
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check<ppln::robots::Xarm7>(float* sphere_pos, int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool has_collision=false;
    
        for (int i = thread_ind; i < XARM7_SPHERE_COUNT-XARM7_SPHERE_COUNT%4; i += 4){
            // sphere i, robot batch_ind (16 robots)
            if (i > 0 && (joint_in_collision[20*batch_ind + xarm7_sphere_to_joint[i]] & 1) && 
                sphere_environment_in_collision(
                    env,
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                    xarm7_spheres_array[i].w
                )
            ) {
                has_collision=true;
            } 
            if (warp_any_full_mask(has_collision)) return false;
        }
        int i=XARM7_SPHERE_COUNT-1-thread_ind;
        if (i > 0 && (joint_in_collision[20*batch_ind + xarm7_sphere_to_joint[i]] & 1) && 
            sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                xarm7_spheres_array[i].w
            )
        ) {
            return false;
        } 
        return true;
    }

    template <>
    __device__ bool self_collision_check_full<ppln::robots::Xarm7>(float* sphere_pos, int* joint_in_collision, const int tid){
        (void)joint_in_collision;
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool has_collision = false;
    
        for (int i = thread_ind; i < XARM7_SELF_CC_RANGE_COUNT; i += 4) {
            if (warp_any_active_mask(has_collision)) return false;
            int sphere_1_ind = xarm7_self_cc_ranges[i][0];
            float sphere_1[3] = {
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            for (int j = xarm7_self_cc_ranges[i][1]; j <= xarm7_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], xarm7_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], xarm7_spheres_array[j].w
                )){
                    has_collision = true;
                }
            }
        }
        return !has_collision;
    
    }

    template <>
    __device__ bool env_collision_check_full<ppln::robots::Xarm7>(float* sphere_pos, int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        (void)joint_in_collision;
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool has_collision=false;
    
        for (int i = thread_ind; i < XARM7_SPHERE_COUNT-XARM7_SPHERE_COUNT%4; i += 4){
            // sphere i, robot batch_ind (16 robots)
            if (i > 0 &&
                sphere_environment_in_collision(
                    env,
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                    xarm7_spheres_array[i].w
                )
            ) {
                has_collision=true;
            } 
            if (warp_any_full_mask(has_collision)) return false;
        }
        int i=XARM7_SPHERE_COUNT-1-thread_ind;
        if (i > 0 &&
            sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                xarm7_spheres_array[i].w
            )
        ) {
            return false;
        } 
        return true;
    }
    }
    
