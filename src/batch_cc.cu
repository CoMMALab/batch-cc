/*
Multi environment batch collision checker.
*/

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/Planners.hh"
#include "src/pRRTC_settings.hh"
#include "src/utils.cuh"
#include "batch_cc.hh"
#include "robots/panda.cuh"
#include "robots/xarm7.cuh"


#include <float.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        throw std::runtime_error(std::string("CUDA error: ") +        \
            cudaGetErrorString(_e) + " at " + __FILE__ + ":" +        \
            std::to_string(__LINE__));                                \
    }                                                                 \
} while (0)

namespace batch_cc {

    static int g_max_blocks = 0;

    void set_max_blocks(int max_blocks) {
        g_max_blocks = max_blocks;
    }

    using EnvF = ppln::collision::Environment<float>;
    using ppln::collision::Sphere;
    using ppln::collision::Capsule;
    using ppln::collision::Cylinder;
    using ppln::collision::Cuboid;

    static inline void ensure_ptr_count_consistent(const EnvF& e) {
        auto ok = true;
        ok &= (e.num_spheres            == 0) || (e.spheres            != nullptr);
        ok &= (e.num_capsules           == 0) || (e.capsules           != nullptr);
        ok &= (e.num_z_aligned_capsules == 0) || (e.z_aligned_capsules != nullptr);
        ok &= (e.num_cylinders          == 0) || (e.cylinders          != nullptr);
        ok &= (e.num_cuboids            == 0) || (e.cuboids            != nullptr);
        ok &= (e.num_z_aligned_cuboids  == 0) || (e.z_aligned_cuboids  != nullptr);
        if (!ok) {
            throw std::runtime_error("Host environment has positive counts with null pointers.");
        }
    }

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };

    void __device__ shuffle_array(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    template <typename Robot>
    __global__ void
    batch_cc_kernel(ppln::collision::Environment<float>* envs, float* edges, int num_envs, int num_edges, uint8_t *cc_result, int resolution)
    {
       
        constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bdim = (blockDim.x / 4);
        const int batch_idx = tid / 4;
        const int col_idx = tid % 4;

        __align__(16) __shared__ volatile float sphere_pos[80 * G_BATCH_SIZE * 3]; // ~assuming max 60 spheres with granularity 32, each has x y z coordinates
        // __align__(16) __shared__ volatile float sphere_pos_approx[10 * G_BATCH_SIZE * 3]; // ~assuming 10 spheres with granularity 32, each has x y z coordinates
        __align__(16) __shared__ volatile int link_CC[G_BATCH_SIZE * 20]; //assuming max granularity 32, max number of links 20
        __align__(16) __shared__ float T[G_BATCH_SIZE * 1 * 16]; // 32 robots x 1x4x4 transform matrix

        const int total_pairs = num_envs * num_edges;
        for (int pair_idx = blockIdx.x; pair_idx < total_pairs; pair_idx += gridDim.x) {
            // each block handles one (edge, environment) pair per loop iteration
            const int env_idx = pair_idx % num_envs;
            const int edge_idx = pair_idx / num_envs;

            ppln::collision::Environment<float>* env = &envs[env_idx];
            __shared__ float edge_start[dim];
            __shared__ float edge_end[dim];
            __shared__ float delta[dim];
            __shared__ unsigned int local_cc_result;
            __shared__ unsigned int any_approx_env_collision;
            __shared__ unsigned int any_approx_self_collision;
            __shared__ int n;
            float config[dim];
            if (tid < dim) {
                // edge_start[tid] = edges[0][tid][edge_idx];
                // edge_end[tid] = edges[1][tid][edge_idx];
                edge_start[tid] = edges[edge_idx * (dim * 2) + 0 * dim + tid];
                edge_end[tid] = edges[edge_idx * (dim * 2) + 1 * dim + tid];
            }
            __syncthreads();
            if (tid == 0) {
                float dist = sqrt(device_utils::sq_l2_dist(edge_start, edge_end, dim));
                n = max(ceil((dist / (float) bdim) * resolution), 1.0f);
                local_cc_result = 0;
                // printf("n: %d, dist: %f\n", n, dist);
            }
            __syncthreads();
            if (tid < dim) {
                // float dist = sqrt(device_utils::sq_l2_dist(edge_start, edge_end, dim));
                // n = max(ceil((dist / (float) bdim) * resolution), 1.0f);
                // local_cc_result = 0;
                delta[tid] = (edge_end[tid] - edge_start[tid]) / (float) (bdim * n);
            }
            __syncthreads();

            # pragma unroll
            for (int j = 0; j < dim; j++) {
                config[j] = edge_start[j] + delta[j] * (batch_idx * n);
            }
            for (int i = 0; i < n; i++) {            
                // ppln::collision::fkcc<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, sphere_pos_approx, link_CC, T, &local_cc_result);
                ppln::collision::fkcc_single_buffer<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, link_CC, T, &local_cc_result, &any_approx_env_collision, &any_approx_self_collision);
                if (local_cc_result) break;
                # pragma unroll
                for (int j = 0; j < dim; j++) {
                    config[j] += delta[j];
                }
            }
            // check end point
            if (!local_cc_result) {
                # pragma unroll
                for (int j = 0; j < dim; j++) {
                    config[j] = edge_end[j];
                }
                ppln::collision::fkcc_single_buffer<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, link_CC, T, &local_cc_result, &any_approx_env_collision, &any_approx_self_collision);
            }
            if (tid == 0) {
                cc_result[edge_idx * num_envs + env_idx] = local_cc_result ? 1 : 0;
            }
            __syncthreads();
        }
    }

    void setup_environments_on_device_bulk(std::vector<ppln::collision::Environment<float>>& h_envs, EnvF*& d_envs, void*& d_blob) {
        // create one blob that holds all the environments
        // find blob size by iterating over all environments and adding up the sizes of the blobs
        size_t blob_size = 0;
        std::vector<size_t> env_blob_offsets(h_envs.size());
        std::vector<EnvF> h_shadow_envs(h_envs.size());
        for (size_t i = 0; i < h_envs.size(); ++i) {
            env_blob_offsets[i] = blob_size;
            blob_size += h_envs[i].num_spheres * sizeof(Sphere<float>) +
                         h_envs[i].num_capsules * sizeof(Capsule<float>) +
                         h_envs[i].num_z_aligned_capsules * sizeof(Capsule<float>) +
                         h_envs[i].num_cylinders * sizeof(Cylinder<float>) +
                         h_envs[i].num_cuboids * sizeof(Cuboid<float>) +
                         h_envs[i].num_z_aligned_cuboids * sizeof(Cuboid<float>);
            h_shadow_envs[i].owns_memory = false;
        }


        void *h_blob = nullptr;
        cudaMallocHost(&h_blob, blob_size);
        cudaMalloc(&d_blob, blob_size);
        char *h_base = static_cast<char*>(h_blob);
        char *d_base = static_cast<char*>(d_blob);
        size_t off = 0;
        for (size_t i = 0; i < h_envs.size(); ++i) {
            if (h_envs[i].num_spheres > 0) {
                std::memcpy(h_base + off, h_envs[i].spheres, h_envs[i].num_spheres * sizeof(Sphere<float>));
                h_shadow_envs[i].spheres = reinterpret_cast<Sphere<float>*>(d_base + off);
                h_shadow_envs[i].num_spheres = h_envs[i].num_spheres;
                off += h_envs[i].num_spheres * sizeof(Sphere<float>);
            }
            if (h_envs[i].num_capsules > 0) {
                std::memcpy(h_base + off, h_envs[i].capsules, h_envs[i].num_capsules * sizeof(Capsule<float>));
                h_shadow_envs[i].capsules = reinterpret_cast<Capsule<float>*>(d_base + off);
                h_shadow_envs[i].num_capsules = h_envs[i].num_capsules;
                off += h_envs[i].num_capsules * sizeof(Capsule<float>);
            }
            if (h_envs[i].num_z_aligned_capsules > 0) {
                std::memcpy(h_base + off, h_envs[i].z_aligned_capsules, h_envs[i].num_z_aligned_capsules * sizeof(Capsule<float>));
                h_shadow_envs[i].z_aligned_capsules = reinterpret_cast<Capsule<float>*>(d_base + off);
                h_shadow_envs[i].num_z_aligned_capsules = h_envs[i].num_z_aligned_capsules;
                off += h_envs[i].num_z_aligned_capsules * sizeof(Capsule<float>);
            }
            if (h_envs[i].num_cylinders > 0) {
                std::memcpy(h_base + off, h_envs[i].cylinders, h_envs[i].num_cylinders * sizeof(Cylinder<float>));
                h_shadow_envs[i].cylinders = reinterpret_cast<Cylinder<float>*>(d_base + off);
                h_shadow_envs[i].num_cylinders = h_envs[i].num_cylinders;
                off += h_envs[i].num_cylinders * sizeof(Cylinder<float>);
            }
            if (h_envs[i].num_cuboids > 0) {
                std::memcpy(h_base + off, h_envs[i].cuboids, h_envs[i].num_cuboids * sizeof(Cuboid<float>));
                h_shadow_envs[i].cuboids = reinterpret_cast<Cuboid<float>*>(d_base + off);
                h_shadow_envs[i].num_cuboids = h_envs[i].num_cuboids;
                off += h_envs[i].num_cuboids * sizeof(Cuboid<float>);
            }
            if (h_envs[i].num_z_aligned_cuboids > 0) {
                std::memcpy(h_base + off, h_envs[i].z_aligned_cuboids, h_envs[i].num_z_aligned_cuboids * sizeof(Cuboid<float>));
                h_shadow_envs[i].z_aligned_cuboids = reinterpret_cast<Cuboid<float>*>(d_base + off);
                h_shadow_envs[i].num_z_aligned_cuboids = h_envs[i].num_z_aligned_cuboids;
                off += h_envs[i].num_z_aligned_cuboids * sizeof(Cuboid<float>);
            }
        }

        EnvF* d_env = nullptr;
        cudaMalloc(&d_env, sizeof(EnvF) * h_envs.size());
        cudaMemcpy(d_env, h_shadow_envs.data(), sizeof(EnvF) * h_envs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_blob, h_blob, blob_size, cudaMemcpyHostToDevice);
        cudaFreeHost(h_blob);
        d_envs = d_env;
        d_blob = d_blob;
    }


    template <typename Robot>
    void batch_cc(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename Robot::Configuration, 2>>& edges, int resolution, std::vector<uint8_t>& results) {
        auto setup_start_time = std::chrono::steady_clock::now();

        EnvF* d_envs = nullptr;
        void* d_blob = nullptr;
        setup_environments_on_device_bulk(h_envs, d_envs, d_blob);

        int num_envs = h_envs.size();
        int num_edges = edges.size();
        int total_pairs = num_envs * num_edges;
        int max_blocks = g_max_blocks > 0 ? g_max_blocks : total_pairs;
        int num_blocks = std::min(total_pairs, max_blocks);
        if (num_blocks <= 0 && total_pairs > 0) {
            num_blocks = 1;
        }
        int num_threads = G_BATCH_SIZE * 4;
        auto env_setup_ns = get_elapsed_nanoseconds(setup_start_time);
        std::cout << "Environments Setup time: " << env_setup_ns / 1'000'000'000.0 << " s" << std::endl;
        uint8_t *d_cc_result;
        cudaMalloc(&d_cc_result, sizeof(uint8_t) * num_envs * num_edges);

        float *d_edges;
        size_t edges_size = edges.size() * Robot::dimension * 2 * sizeof(float);
        cudaMalloc(&d_edges, edges_size);
        cudaMemcpy(d_edges, edges.data(), edges_size, cudaMemcpyHostToDevice);

        auto setup_ns = get_elapsed_nanoseconds(setup_start_time);
        std::cout << "Setup time: " << setup_ns / 1'000'000'000.0 << " s" << std::endl;
        cudaCheckError(cudaGetLastError());
        auto kernel_start_time = std::chrono::steady_clock::now();
        batch_cc_kernel<Robot><<<num_blocks, num_threads>>>(d_envs, d_edges, num_envs, num_edges, d_cc_result, resolution);
        cudaDeviceSynchronize();
        auto kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        std::cout << "Kernel time: " << kernel_ns << " ns" << std::endl;
        int edges_checked = num_envs * num_edges;
        std::cout << "Edges checked: " << edges_checked << std::endl;
        double throughput = edges_checked / (kernel_ns / 1e9);
        std::cout << "Throughput: " << throughput << " edges/s" << std::endl;

        // Create a temporary buffer for the results
        auto cleanup_start_time = std::chrono::steady_clock::now();
        cudaMemcpy(results.data(), d_cc_result, sizeof(uint8_t) * num_envs * num_edges, cudaMemcpyDeviceToHost);
        
        cudaCheckError(cudaGetLastError());
        cudaFree(d_cc_result);
        cudaFree(d_edges);
        cudaFree(d_envs);
        cudaFree(d_blob);
        auto cleanup_ns = get_elapsed_nanoseconds(cleanup_start_time);
        std::cout << "Cleanup time: " << cleanup_ns << " ns" << std::endl;
    }

    template void batch_cc<typename ppln::robots::Panda>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Panda::Configuration, 2>>& edges, int resolution, std::vector<uint8_t>& results);
    template void batch_cc<typename ppln::robots::Xarm7>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Xarm7::Configuration, 2>>& edges, int resolution, std::vector<uint8_t>& results);
    // template void batch_cc<typename ppln::robots::Fetch>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Fetch::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
    // template void batch_cc<typename ppln::robots::Baxter>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Baxter::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
} // namespace batch_cc
