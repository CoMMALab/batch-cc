/**********************************************************************
*  batch_cc_two_stage.cu                                              *
*                                                                     *
*  Two–stage batch collision checker                                   *
*    Stage-1 :  prune (mid-point test, 32 envs per block)             *
*    Stage-2 :  full discretised edge check on surviving pairs        *
**********************************************************************/

// this version checks 3 points during the prune stage
// (0.0, 0.5, 0.75) and uses a compacted list of pairs for the fine stage
// this version checks midpoint and then does no compaction
// the fine kernel just early exits if the edge is already in collision

// #pragma once

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/Planners.hh"
#include "src/pRRTC_settings.hh"
#include "src/utils.cuh"
#include "batch_cc.hh"

#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>     // std::accumulate

namespace batch_cc
{
/*-----------------------------------------------------------*
 |  0.  helpers                                              |
 *-----------------------------------------------------------*/
inline __host__ __device__ constexpr int div_up(int a, int b)
{ return (a + b - 1) / b; }

inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env, 
    const ppln::collision::Environment<float> &h_env) {
    // First allocate the environment struct
    cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));

    // Initialize struct to zeros first
    cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

    // Handle each primitive type separately
    if (h_env.num_spheres > 0) {
    // Allocate and copy spheres array
    ppln::collision::Sphere<float> *d_spheres;
    cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
    cudaMemcpy(d_spheres, h_env.spheres, 
    sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres, 
    cudaMemcpyHostToDevice);

    // Update the struct fields directly
    cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), 
    cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int), 
    cudaMemcpyHostToDevice);
    }

    if (h_env.num_capsules > 0) {
    ppln::collision::Capsule<float> *d_capsules;
    cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
    cudaMemcpy(d_capsules, h_env.capsules,
    sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,
    cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),
    cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),
    cudaMemcpyHostToDevice);
    }

    // Repeat for each primitive type...
    if (h_env.num_z_aligned_capsules > 0) {
    ppln::collision::Capsule<float> *d_z_capsules;
    cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
    cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,
    sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,
    cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),
    cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),
    cudaMemcpyHostToDevice);
    }

    if (h_env.num_cylinders > 0) {
    ppln::collision::Cylinder<float> *d_cylinders;
    cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
    cudaMemcpy(d_cylinders, h_env.cylinders,
    sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,
    cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),
    cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),
    cudaMemcpyHostToDevice);
    }

    if (h_env.num_cuboids > 0) {
    ppln::collision::Cuboid<float> *d_cuboids;
    cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
    cudaMemcpy(d_cuboids, h_env.cuboids,
    sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,
    cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),
    cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),
    cudaMemcpyHostToDevice);
    }

    if (h_env.num_z_aligned_cuboids > 0) {
    ppln::collision::Cuboid<float> *d_z_cuboids;
    cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
    cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,
    sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,
    cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),
    cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),
    cudaMemcpyHostToDevice);
    }
}

inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, 
    const ppln::collision::Environment<float> &h_env) {
    // Get the pointers from device struct before freeing
    ppln::collision::Sphere<float> *d_spheres = nullptr;
    ppln::collision::Capsule<float> *d_capsules = nullptr;
    ppln::collision::Capsule<float> *d_z_capsules = nullptr;
    ppln::collision::Cylinder<float> *d_cylinders = nullptr;
    ppln::collision::Cuboid<float> *d_cuboids = nullptr;
    ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

    // Copy each pointer from device memory
    if (h_env.num_spheres > 0) {
    cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
    cudaFree(d_spheres);
    }

    if (h_env.num_capsules > 0) {
    cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
    cudaFree(d_capsules);
    }

    if (h_env.num_z_aligned_capsules > 0) {
    cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
    cudaFree(d_z_capsules);
    }

    if (h_env.num_cylinders > 0) {
    cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
    cudaFree(d_cylinders);
    }

    if (h_env.num_cuboids > 0) {
    cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
    cudaFree(d_cuboids);
    }

    if (h_env.num_z_aligned_cuboids > 0) {
    cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
    cudaFree(d_z_cuboids);
    }

    // Finally free the environment struct itself
    cudaFree(d_env);
}


__device__ int write_index = 0;
/*-----------------------------------------------------------*
 |  1.  Stage-1 prune kernel                                 |
 *-----------------------------------------------------------*/
template <typename Robot>
__global__ void prune_kernel(
    ppln::collision::Environment<float>** envs,
    float* edges[2][Robot::dimension],
    int num_envs,
    int num_edges,
    bool* cc_result_full
)
{
    constexpr int dim = Robot::dimension;

    /* grid layout -------------------------------------------------- *
     *  blockIdx.x  -> edge id                                        *
     *  blockIdx.y  -> stripe of 32 environments                      *
     *  threadIdx.x -> 0‥31, one environment inside the stripe        */
    const int edge_idx = blockIdx.x;
    const int env_idx  = blockIdx.y * 32 + threadIdx.x;

    if (edge_idx >= num_edges || env_idx >= num_envs) return;

    /* fetch edge endpoints (SoA) into registers */
    __shared__ float start[dim], delta[dim], cfg[dim];

    if (threadIdx.x < dim) {
        start[threadIdx.x] = edges[0][threadIdx.x][edge_idx];
        delta[threadIdx.x] = edges[1][threadIdx.x][edge_idx] - start[threadIdx.x];
        cfg[threadIdx.x] = start[threadIdx.x] + delta[threadIdx.x] * 0.5f;
    }
    __syncthreads();

    /* midpoint collision test */
    bool coll = not ppln::collision::fkcc<Robot>(cfg, envs[env_idx], /*lane=*/0);

    /* write flag (1 = collided already -> DISCARD) */
    cc_result_full[edge_idx * num_envs + env_idx] = coll;
}

/*-----------------------------------------------------------*
 |  2.  Stage-2 fine kernel (mostly your old one)            |
 *-----------------------------------------------------------*/
template <typename Robot>
__global__ void fine_kernel(
        ppln::collision::Environment<float>** envs,
        float* edges[2][Robot::dimension],
        int num_pairs,               
        bool* cc_result_full,         
        int num_envs,                 
        int resolution)
{
    constexpr int dim = Robot::dimension;
    const int tid = threadIdx.x;
    const int pair_idx = blockIdx.x * blockDim.x + blockIdx.y;
    if (pair_idx >= num_pairs) return;

    /* map block -> (edge, env) */
    const int edge_idx = blockIdx.x;
    const int env_idx  = blockIdx.y;

    if (cc_result_full[edge_idx * num_envs + env_idx]) return;

    ppln::collision::Environment<float>* env = envs[env_idx];

    /* shared memory for this block */
    __shared__ float edge_start[dim];
    __shared__ float edge_end  [dim];
    __shared__ float delta     [dim];
    __shared__ bool  local_cc_result;
    __shared__ int   n;

    /* load endpoints ------------------------------------------------*/
    if (tid < dim) {
        edge_start[tid] = edges[0][tid][edge_idx];
        edge_end  [tid] = edges[1][tid][edge_idx];
    }
    __syncthreads();

    /* discretisation count per lane -------------------------------- */
    if (tid == 0) {
        float dist = sqrt(device_utils::sq_l2_dist(edge_start, edge_end, dim));
        n = max(ceil((dist / (float) blockDim.x) * resolution), 1.0f);
        local_cc_result = false;
    }
    __syncthreads();

    if (tid < dim)
        delta[tid] = (edge_end[tid] - edge_start[tid]) / (float)(blockDim.x * n);
    __syncthreads();

    /* first configuration checked by each lane */
    float cfg[dim];
    #pragma unroll
    for (int d = 0; d < dim; ++d)
        cfg[d] = edge_start[d] + delta[d] * (tid * n);

    /* loop over n samples per lane ----------------------------------*/
    for (int i = 0; i < n; ++i) {
        bool in_collision = not ppln::collision::fkcc<Robot>(cfg, env, tid);
        local_cc_result = __any_sync(0xffffffff, in_collision);
        if (local_cc_result) break;

        #pragma unroll
        for (int d = 0; d < dim; ++d) cfg[d] += delta[d];
    }

    /* write final result */
    if (tid == 0)
        cc_result_full[edge_idx * num_envs + env_idx] = local_cc_result;
}




/*-----------------------------------------------------------*
 |  3.  Host-side entry – two-stage pipeline                 |
 *-----------------------------------------------------------*/
template <typename Robot>
void batch_cc(std::vector<ppln::collision::Environment<float>>& h_envs,
              std::vector<std::array<typename Robot::Configuration,2>>& edges,
              int resolution,
              std::vector<bool>& results)
{
    /* ---------- device-side environments -------------------------- */
    const int num_envs  = (int)h_envs.size();
    const int num_edges = (int)edges.size();

    std::vector<ppln::collision::Environment<float>*> d_envs(h_envs.size());
    for (size_t i = 0; i < h_envs.size(); ++i)
        setup_environment_on_device(d_envs[i], h_envs[i]);

    ppln::collision::Environment<float>** d_envs_ptr;
    cudaMalloc(&d_envs_ptr, sizeof(ppln::collision::Environment<float>*) * num_envs);
    cudaMemcpy(d_envs_ptr, d_envs.data(),
               sizeof(ppln::collision::Environment<float>*) * num_envs,
               cudaMemcpyHostToDevice);

    /* ---------- SoA of edges -------------------------------------- */
    float* d_edges[2][Robot::dimension];
    for (int d = 0; d < Robot::dimension; ++d) {
        cudaMalloc(&d_edges[0][d], sizeof(float) * num_edges);
        cudaMalloc(&d_edges[1][d], sizeof(float) * num_edges);
        for (int e = 0; e < num_edges; ++e) {
            cudaMemcpy(d_edges[0][d] + e, &edges[e][0][d],
                       sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_edges[1][d] + e, &edges[e][1][d],
                       sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    float* (*d_edges_ptr)[Robot::dimension];
    cudaMalloc(&d_edges_ptr, sizeof(float*) * 2 * Robot::dimension);
    cudaMemcpy(d_edges_ptr, d_edges,
               sizeof(float*) * 2 * Robot::dimension,
               cudaMemcpyHostToDevice);
    
    // allocate final result array and work pairs array
    const int total_pairs = num_edges * num_envs;
    bool* d_cc_full;
    cudaMalloc(&d_cc_full, sizeof(bool) * total_pairs);
    cudaMemset(d_cc_full, 0, sizeof(bool) * total_pairs);


    const int stripe_cnt = div_up(num_envs, 32);
    dim3 block1(32);
    dim3 grid1(num_edges, stripe_cnt);

    auto start_time = std::chrono::steady_clock::now();
    auto prune_start_time = std::chrono::steady_clock::now();
    
    /* prune and compact 1 */
    prune_kernel<Robot><<<grid1, block1>>>(
        d_envs_ptr, d_edges_ptr,
        num_envs, num_edges,
        d_cc_full
    );
    
    auto prune_time = get_elapsed_nanoseconds(prune_start_time);
    /* ---------- stage-2 fine kernel ------------------------------- */
    auto fine_start_time = std::chrono::steady_clock::now();
    int threads2 = 32;
    dim3 blocks2(num_edges, num_envs);
    fine_kernel<Robot><<<blocks2, threads2>>>(
        d_envs_ptr,
        d_edges_ptr,
        total_pairs,
        d_cc_full,
        num_envs,
        resolution
    );

    cudaDeviceSynchronize();
    auto fine_time = get_elapsed_nanoseconds(fine_start_time);
    auto total_time = get_elapsed_nanoseconds(start_time);

    cudaCheckError(cudaGetLastError());


    std::cout << "Total time: " << total_time << " ns" << std::endl;
    std::cout << "Prune time: " << prune_time << " ns" << std::endl;
    std::cout << "Fine time: " << fine_time << " ns" << std::endl;
    std::cout << "Edges checked: " << total_pairs << std::endl;
    // std::cout << "Edges remaining: " << num_remaining << std::endl;
    // std::cout << "Edges pruned: " << total_pairs - num_remaining << std::endl;
    double throughput = total_pairs / (total_time / 1e9);
    std::cout << "Throughput: " << throughput << " edges/s" << std::endl;

    /* ---------- copy results back to host ------------------------- */
    // Create a temporary buffer for the results
    bool* h_cc_result = new bool[num_envs * num_edges];
    cudaMemcpy(h_cc_result, d_cc_full, sizeof(bool) * num_envs * num_edges, cudaMemcpyDeviceToHost);
    
    // Copy from temporary buffer to vector<bool>
    for (int i = 0; i < num_envs * num_edges; ++i) {
        results[i] = h_cc_result[i];
    }
    delete[] h_cc_result;

    /* ---------- clean up ----------------------------------------- */
    cudaFree(d_cc_full);

    for (int d = 0; d < Robot::dimension; ++d) {
        cudaFree(d_edges[0][d]);
        cudaFree(d_edges[1][d]);
    }
    cudaFree(d_edges_ptr);

    for (size_t i = 0; i < h_envs.size(); ++i)
        cleanup_environment_on_device(d_envs[i], h_envs[i]);
    cudaFree(d_envs_ptr);
}

/*-----------------------------------------------------------*
 |  4.  explicit template instantiations                     |
 *-----------------------------------------------------------*/
template void batch_cc<typename ppln::robots::Panda>(
        std::vector<ppln::collision::Environment<float>>&,
        std::vector<std::array<typename ppln::robots::Panda::Configuration,2>>&,
        int,
        std::vector<bool>&);

template void batch_cc<typename ppln::robots::Fetch>(
        std::vector<ppln::collision::Environment<float>>&,
        std::vector<std::array<typename ppln::robots::Fetch::Configuration,2>>&,
        int,
        std::vector<bool>&);

template void batch_cc<typename ppln::robots::Baxter>(
        std::vector<ppln::collision::Environment<float>>&,
        std::vector<std::array<typename ppln::robots::Baxter::Configuration,2>>&,
        int,
        std::vector<bool>&);

} // namespace batch_cc
