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

#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/Planners.hh"
#include "src/pRRTC_settings.hh"
#include "src/utils.cuh"
#include "batch_cc.hh"
#include "robots/panda.cuh"
#include "robots/xarm7.cuh"


#include <float.h>

namespace cg = cooperative_groups;

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
    static TwoPhaseMode g_two_phase_mode = TwoPhaseMode::FullOnly;

    void set_max_blocks(int max_blocks) {
        g_max_blocks = max_blocks;
    }

    void set_two_phase_mode(TwoPhaseMode mode) {
        g_two_phase_mode = mode;
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

    __global__ void init_indices(int *indices, int count) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            indices[idx] = idx;
        }
    }

    template <typename Robot>
    __global__ void
    batch_cc_kernel_approx(ppln::collision::Environment<float>* envs, float* edges, int num_envs, int num_edges,
                           uint8_t *cc_result, uint8_t *candidate_flags, int resolution)
    {
        constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bdim = (blockDim.x / 4);
        const int batch_idx = tid / 4;
        const int thread_ind = tid % 4;

        auto cta = cg::this_thread_block();
        auto tile = cg::tiled_partition<4>(cta);

        __align__(16) __shared__ float sphere_pos[MAX_SPHERE_COUNT * G_BATCH_SIZE * 3];
        __align__(16) __shared__ int link_CC[G_BATCH_SIZE * 20];
        __align__(16) __shared__ float T[G_BATCH_SIZE * 1 * 16];

        const int total_pairs = num_envs * num_edges;
        for (int pair_idx = blockIdx.x; pair_idx < total_pairs; pair_idx += gridDim.x) {
            const int env_idx = pair_idx % num_envs;
            const int edge_idx = pair_idx / num_envs;
            ppln::collision::Environment<float>* env = &envs[env_idx];

            __shared__ float edge_start[dim];
            __shared__ float edge_end[dim];
            __shared__ float delta[dim];
            __shared__ unsigned int local_has_candidate;
            __shared__ int n;
            float config[dim];

            if (tid < dim) {
                edge_start[tid] = edges[edge_idx * (dim * 2) + 0 * dim + tid];
                edge_end[tid] = edges[edge_idx * (dim * 2) + 1 * dim + tid];
            }
            __syncthreads();
            if (tid == 0) {
                float dist = sqrt(device_utils::sq_l2_dist(edge_start, edge_end, dim));
                n = max(ceil((dist / (float) bdim) * resolution), 1.0f);
                local_has_candidate = 0;
            }
            __syncthreads();
            if (tid < dim) {
                delta[tid] = (edge_end[tid] - edge_start[tid]) / (float) (bdim * n);
            }
            __syncthreads();

            # pragma unroll
            for (int j = 0; j < dim; j++) {
                config[j] = edge_start[j] + delta[j] * (batch_idx * n);
            }
            for (int i = 0; i < n; i++) {
                for (int k = thread_ind; k < 20; k += 4) {
                    link_CC[20 * batch_idx + k] = 0;
                }
                tile.sync();

                ppln::collision::fk_approx<Robot>(config, sphere_pos, T, tid);
                tile.sync();

                bool approx_env_collision =
                    not ppln::collision::env_collision_check_approx<Robot>(sphere_pos, link_CC, env, tid);
                bool approx_self_collision =
                    not ppln::collision::self_collision_check_approx<Robot>(sphere_pos, link_CC, tid);

                bool any_collision = tile.any(approx_env_collision) || tile.any(approx_self_collision);
                if (tile.thread_rank() == 0 && any_collision) {
                    atomicOr(&local_has_candidate, 1u);
                }
                __syncthreads();
                if (local_has_candidate) {
                    break;
                }
                # pragma unroll
                for (int j = 0; j < dim; j++) {
                    config[j] += delta[j];
                }
            }
            if (!local_has_candidate) {
                # pragma unroll
                for (int j = 0; j < dim; j++) {
                    config[j] = edge_end[j];
                }
                for (int k = thread_ind; k < 20; k += 4) {
                    link_CC[20 * batch_idx + k] = 0;
                }
                tile.sync();

                ppln::collision::fk_approx<Robot>(config, sphere_pos, T, tid);
                tile.sync();

                bool approx_env_collision =
                    not ppln::collision::env_collision_check_approx<Robot>(sphere_pos, link_CC, env, tid);
                bool approx_self_collision =
                    not ppln::collision::self_collision_check_approx<Robot>(sphere_pos, link_CC, tid);
                bool any_collision = tile.any(approx_env_collision) || tile.any(approx_self_collision);
                if (tile.thread_rank() == 0 && any_collision) {
                    atomicOr(&local_has_candidate, 1u);
                }
                __syncthreads();
            }

            if (tid == 0) {
                cc_result[edge_idx * num_envs + env_idx] = 0;
                candidate_flags[pair_idx] = local_has_candidate ? 1 : 0;
            }
            __syncthreads();
        }
    }

    template <typename Robot, bool ApproxGated>
    __global__ void
    batch_cc_kernel_full_candidates(ppln::collision::Environment<float>* envs, float* edges, int num_envs, int num_edges,
                                    const int *candidate_indices, int num_candidates, uint8_t *cc_result, int resolution)
    {
        constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bdim = (blockDim.x / 4);
        const int batch_idx = tid / 4;

        __align__(16) __shared__ float sphere_pos[MAX_SPHERE_COUNT * G_BATCH_SIZE * 3];
        __align__(16) __shared__ int link_CC[G_BATCH_SIZE * 20];
        __align__(16) __shared__ float T[G_BATCH_SIZE * 1 * 16];

        for (int candidate_idx = blockIdx.x; candidate_idx < num_candidates; candidate_idx += gridDim.x) {
            int pair_idx = candidate_indices[candidate_idx];
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
                edge_start[tid] = edges[edge_idx * (dim * 2) + 0 * dim + tid];
                edge_end[tid] = edges[edge_idx * (dim * 2) + 1 * dim + tid];
            }
            __syncthreads();
            if (tid == 0) {
                float dist = sqrt(device_utils::sq_l2_dist(edge_start, edge_end, dim));
                n = max(ceil((dist / (float) bdim) * resolution), 1.0f);
                local_cc_result = 0;
            }
            __syncthreads();
            if (tid < dim) {
                delta[tid] = (edge_end[tid] - edge_start[tid]) / (float) (bdim * n);
            }
            __syncthreads();

            # pragma unroll
            for (int j = 0; j < dim; j++) {
                config[j] = edge_start[j] + delta[j] * (batch_idx * n);
            }
            for (int i = 0; i < n; i++) {
                if constexpr (ApproxGated) {
                    ppln::collision::fkcc_single_buffer<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, link_CC, T, &local_cc_result, &any_approx_env_collision, &any_approx_self_collision);
                } else {
                    ppln::collision::fkcc_detailed_only<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, sphere_pos, link_CC, T, &local_cc_result);
                }
                if (local_cc_result) break;
                # pragma unroll
                for (int j = 0; j < dim; j++) {
                    config[j] += delta[j];
                }
            }
            if (!local_cc_result) {
                # pragma unroll
                for (int j = 0; j < dim; j++) {
                    config[j] = edge_end[j];
                }
                if constexpr (ApproxGated) {
                    ppln::collision::fkcc_single_buffer<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, link_CC, T, &local_cc_result, &any_approx_env_collision, &any_approx_self_collision);
                } else {
                    ppln::collision::fkcc_detailed_only<Robot>(config, env, tid, env_idx, edge_idx, sphere_pos, sphere_pos, link_CC, T, &local_cc_result);
                }
            }
            if (tid == 0) {
                cc_result[edge_idx * num_envs + env_idx] = local_cc_result ? 1 : 0;
            }
            __syncthreads();
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

        __align__(16) __shared__ float sphere_pos[MAX_SPHERE_COUNT * G_BATCH_SIZE * 3]; // max spheres per robot, each has x y z coordinates
        // __align__(16) __shared__ volatile float sphere_pos_approx[10 * G_BATCH_SIZE * 3]; // ~assuming 10 spheres with granularity 32, each has x y z coordinates
        __align__(16) __shared__ int link_CC[G_BATCH_SIZE * 20]; // per-batch link flags
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
        uint8_t *d_candidate_flags = nullptr;
        int *d_indices = nullptr;
        int *d_selected_indices = nullptr;
        int *d_num_selected = nullptr;
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        if (total_pairs > 0) {
            cudaMalloc(&d_candidate_flags, sizeof(uint8_t) * total_pairs);
            cudaMalloc(&d_indices, sizeof(int) * total_pairs);
            cudaMalloc(&d_selected_indices, sizeof(int) * total_pairs);
            cudaMalloc(&d_num_selected, sizeof(int));

            int init_threads = 256;
            int init_blocks = (total_pairs + init_threads - 1) / init_threads;
            init_indices<<<init_blocks, init_threads>>>(d_indices, total_pairs);

            batch_cc_kernel_approx<Robot><<<num_blocks, num_threads>>>(d_envs, d_edges, num_envs, num_edges, d_cc_result, d_candidate_flags, resolution);

            cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_indices, d_candidate_flags, d_selected_indices, d_num_selected, total_pairs);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_indices, d_candidate_flags, d_selected_indices, d_num_selected, total_pairs);

            int h_num_selected = 0;
            cudaMemcpy(&h_num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_num_selected > 0) {
                int num_blocks_full = std::min(h_num_selected, max_blocks);
                if (num_blocks_full <= 0) {
                    num_blocks_full = 1;
                }
                if (g_two_phase_mode == TwoPhaseMode::ApproxGated) {
                    batch_cc_kernel_full_candidates<Robot, true><<<num_blocks_full, num_threads>>>(d_envs, d_edges, num_envs, num_edges, d_selected_indices, h_num_selected, d_cc_result, resolution);
                } else {
                    batch_cc_kernel_full_candidates<Robot, false><<<num_blocks_full, num_threads>>>(d_envs, d_edges, num_envs, num_edges, d_selected_indices, h_num_selected, d_cc_result, resolution);
                }
            }
        }
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
        cudaFree(d_candidate_flags);
        cudaFree(d_indices);
        cudaFree(d_selected_indices);
        cudaFree(d_num_selected);
        cudaFree(d_temp_storage);
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
