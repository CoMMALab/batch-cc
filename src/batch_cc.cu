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
    __launch_bounds__(32, 4)
    batch_cc_kernel(ppln::collision::Environment<float>** envs, float* edges, int num_envs, int num_edges, bool *cc_result, int resolution)
    {
        constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int bdim = blockDim.x - 1;
        // each block handles one (edge, environment) pair
        const int env_idx = bid % num_envs;
        const int edge_idx = bid / num_envs;
        if (env_idx >= num_envs || edge_idx >= num_edges) {
            return;
        }
        ppln::collision::Environment<float>* env = envs[env_idx];
        __shared__ float edge_start[dim];
        __shared__ float edge_end[dim];
        __shared__ float delta[dim];
        __shared__ unsigned int local_cc_result;
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
        }
        __syncthreads();
        if (tid < dim) {
            delta[tid] = (edge_end[tid] - edge_start[tid]) / (float) (bdim * n);
        }
        __syncthreads();
        # pragma unroll
        for (int j = 0; j < dim; j++) {
            config[j] = edge_start[j] + delta[j] * (tid * n);
        }
        __syncthreads();
        for (int i = 0; i < n; i++) {
            bool config_in_collision = not ppln::collision::fkcc<Robot>(config, env, tid, env_idx, edge_idx);
            // if (env_idx == 163 && edge_idx == 78) {
            //     printf("Checking config: %f %f %f %f %f %f %f\nin_collision=%d\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config_in_collision?1:0);
            // }
            local_cc_result = __any_sync(0xffffffff, config_in_collision);
            if (local_cc_result) break;
            # pragma unroll
            for (int j = 0; j < dim; j++) {
                config[j] += delta[j];
            }
        }
        if (tid == 0) {
            cc_result[edge_idx * num_envs + env_idx] = local_cc_result ? true : false;
        }
    }

    inline void setup_environment_on_device(EnvF *&d_env,
        const EnvF &h_env,
        std::vector<void*> &d_blobs)
    {
        // static_assert(std::is_trivially_copyable_v<EnvF>,
        // "Environment<T> must be POD/trivially copyable for raw cudaMemcpy.");


        ensure_ptr_count_consistent(h_env);

        // Allocate device struct
        CUDA_CHECK(cudaMalloc(&d_env, sizeof(EnvF)));

        const size_t n_s   = h_env.num_spheres;
        const size_t n_c   = h_env.num_capsules;
        const size_t n_cz  = h_env.num_z_aligned_capsules;
        const size_t n_cyl = h_env.num_cylinders;
        const size_t n_cb  = h_env.num_cuboids;
        const size_t n_cbz = h_env.num_z_aligned_cuboids;

        const size_t sz_s   = n_s   * sizeof(Sphere<float>);
        const size_t sz_c   = n_c   * sizeof(Capsule<float>);
        const size_t sz_cz  = n_cz  * sizeof(Capsule<float>);
        const size_t sz_cyl = n_cyl * sizeof(Cylinder<float>);
        const size_t sz_cb  = n_cb  * sizeof(Cuboid<float>);
        const size_t sz_cbz = n_cbz * sizeof(Cuboid<float>);

        // Compute offsets
        size_t off_s    = 0;
        size_t off_c    = off_s    + sz_s;
        size_t off_cz   = off_c    + sz_c;
        size_t off_cyl  = off_cz   + sz_cz;
        size_t off_cb   = off_cyl  + sz_cyl;
        size_t off_cbz  = off_cb   + sz_cb;
        const size_t blob_size = off_cbz + sz_cbz;


        // Allocate device blob
        void* d_blob = nullptr;
        if (blob_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_blob, blob_size));
        }

        // Prepare host shadow with device pointers
        EnvF h_shadow{};
        h_shadow.num_spheres            = n_s;
        h_shadow.num_capsules           = n_c;
        h_shadow.num_z_aligned_capsules = n_cz;
        h_shadow.num_cylinders          = n_cyl;
        h_shadow.num_cuboids            = n_cb;
        h_shadow.num_z_aligned_cuboids  = n_cbz;
        h_shadow.owns_memory            = false; // This object doesn't own the device memory

        char* base = static_cast<char*>(d_blob);
        h_shadow.spheres            = (n_s   ? reinterpret_cast<Sphere<float>*>( base + off_s   ) : nullptr);
        h_shadow.capsules           = (n_c   ? reinterpret_cast<Capsule<float>*>(base + off_c   ) : nullptr);
        h_shadow.z_aligned_capsules = (n_cz  ? reinterpret_cast<Capsule<float>*>(base + off_cz  ) : nullptr);
        h_shadow.cylinders          = (n_cyl ? reinterpret_cast<Cylinder<float>*>(base + off_cyl) : nullptr);
        h_shadow.cuboids            = (n_cb  ? reinterpret_cast<Cuboid<float>*>( base + off_cb  ) : nullptr);
        h_shadow.z_aligned_cuboids  = (n_cbz ? reinterpret_cast<Cuboid<float>*>( base + off_cbz ) : nullptr);


        // Pack host primitives into a host blob
        if (blob_size > 0) {
        void* h_blob = nullptr;
        cudaError_t pe = cudaMallocHost(&h_blob, blob_size);  // pinned
        if (pe != cudaSuccess) {
        // fallback to pageable to avoid segfault on memcpy to nullptr
        h_blob = std::malloc(blob_size);
        if (!h_blob) {
        CUDA_CHECK(pe); // will throw with the original pinned error
        }
        }

        char* p = static_cast<char*>(h_blob);

        // NOTE: these memcpy read from host pointers in h_env; ensure they are valid
        if (n_s)   { std::memcpy(p + off_s,   h_env.spheres,            sz_s); }
        if (n_c)   { std::memcpy(p + off_c,   h_env.capsules,           sz_c); }
        if (n_cz)  { std::memcpy(p + off_cz,  h_env.z_aligned_capsules, sz_cz); }
        if (n_cyl) { std::memcpy(p + off_cyl, h_env.cylinders,          sz_cyl); }
        if (n_cb)  { std::memcpy(p + off_cb,  h_env.cuboids,            sz_cb); }
        if (n_cbz) { std::memcpy(p + off_cbz, h_env.z_aligned_cuboids,  sz_cbz); }

        // Single bulk copy H2D
        CUDA_CHECK(cudaMemcpy(d_blob, h_blob, blob_size, cudaMemcpyHostToDevice));

        // Free host blob
        if (pe == cudaSuccess) {
        CUDA_CHECK(cudaFreeHost(h_blob));
        } else {
        std::free(h_blob);
        }
        }

        d_blobs.push_back(d_blob);


        // Copy the struct itself
        CUDA_CHECK(cudaMemcpy(d_env, &h_shadow, sizeof(h_shadow), cudaMemcpyHostToDevice));
    }

    inline void cleanup_environment_on_device(EnvF *d_env, void* d_blob) {
        if (d_blob) cudaFree(d_blob);
        if (d_env)  cudaFree(d_env);
    }



    template <typename Robot>
    void batch_cc(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename Robot::Configuration, 2>>& edges, int resolution, std::vector<bool>& results) {
        auto setup_start_time = std::chrono::steady_clock::now();

        std::vector<EnvF*> d_envs;
        d_envs.resize(h_envs.size(), nullptr);

        std::vector<void*> d_blobs;
        d_blobs.reserve(h_envs.size());

        for (size_t i = 0; i < h_envs.size(); ++i) {
            setup_environment_on_device(d_envs[i], h_envs[i], d_blobs);
        }


        int num_envs = h_envs.size();
        int num_edges = edges.size();
        int num_blocks = num_envs * num_edges;
        int num_threads = 32;
        ppln::collision::Environment<float>** d_envs_ptr;
        cudaMalloc(&d_envs_ptr, sizeof(ppln::collision::Environment<float>*) * num_envs);
        cudaMemcpy(d_envs_ptr, d_envs.data(), sizeof(ppln::collision::Environment<float>*) * num_envs, cudaMemcpyHostToDevice);
        auto env_setup_ns = get_elapsed_nanoseconds(setup_start_time);
        std::cout << "Environments Setup time: " << env_setup_ns / 1'000'000'000.0 << " s" << std::endl;
        bool *d_cc_result;
        cudaMalloc(&d_cc_result, sizeof(bool) * num_envs * num_edges);
        // cudaMemset(d_cc_result, 0, sizeof(bool) * num_envs * num_edges);


        // float* d_edges[2][Robot::dimension];        
        // for (int i = 0; i < Robot::dimension; ++i) {
        //     cudaMalloc(&d_edges[0][i], sizeof(float) * num_edges);
        //     cudaMalloc(&d_edges[1][i], sizeof(float) * num_edges);
        //     // Copy the edges to device memory
        //     for (int j = 0; j < num_edges; ++j) {
        //         float start = edges[j][0][i];
        //         float end = edges[j][1][i];
        //         cudaMemcpy(d_edges[0][i] + j, &start, sizeof(float), cudaMemcpyHostToDevice);
        //         cudaMemcpy(d_edges[1][i] + j, &end, sizeof(float), cudaMemcpyHostToDevice);
        //     }
        // }
        // // Allocate memory for the device array of pointers
        // float* (*d_edges_ptr)[Robot::dimension];
        // cudaMalloc(&d_edges_ptr, sizeof(float*) * 2 * Robot::dimension);

        // // Copy the host array of pointers to the device
        // cudaMemcpy(d_edges_ptr, d_edges, sizeof(float*) * 2 * Robot::dimension, cudaMemcpyHostToDevice);

        float *d_edges;
        size_t edges_size = edges.size() * Robot::dimension * 2 * sizeof(float);
        cudaMalloc(&d_edges, edges_size);
        cudaMemcpy(d_edges, edges.data(), edges_size, cudaMemcpyHostToDevice);

        auto setup_ns = get_elapsed_nanoseconds(setup_start_time);
        std::cout << "Setup time: " << setup_ns / 1'000'000'000.0 << " s" << std::endl;
        // std::cout << "here3" << std::endl;
        cudaCheckError(cudaGetLastError());
        // std::cout << "num_envs: " << num_envs << ", num_edges: " << num_edges << "resolution: " << resolution << std::endl;
        auto kernel_start_time = std::chrono::steady_clock::now();
        batch_cc_kernel<Robot><<<num_blocks, num_threads>>>(d_envs_ptr, d_edges, num_envs, num_edges, d_cc_result, resolution);
        cudaDeviceSynchronize();
        auto kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        std::cout << "Kernel time: " << kernel_ns << " ns" << std::endl;
        int edges_checked = num_envs * num_edges;
        std::cout << "Edges checked: " << edges_checked << std::endl;
        double throughput = edges_checked / (kernel_ns / 1e9);
        std::cout << "Throughput: " << throughput << " edges/s" << std::endl;

        // Create a temporary buffer for the results
        auto cleanup_start_time = std::chrono::steady_clock::now();
        bool* h_cc_result = new bool[num_envs * num_edges];
        cudaMemcpy(h_cc_result, d_cc_result, sizeof(bool) * num_envs * num_edges, cudaMemcpyDeviceToHost);
        
        // Copy from temporary buffer to vector<bool>
        for (int i = 0; i < num_envs * num_edges; ++i) {
            results[i] = h_cc_result[i];
        }
        delete[] h_cc_result;

        cudaCheckError(cudaGetLastError());
        // std::cout << "here4" << std::endl;
        for (size_t i = 0; i < h_envs.size(); ++i) {
            cleanup_environment_on_device(d_envs[i], d_blobs[i]);
        }
        cudaFree(d_cc_result);
        // for (int i = 0; i < Robot::dimension; ++i) {
        //     cudaFree(d_edges[0][i]);
        //     cudaFree(d_edges[1][i]);
        // }
        cudaFree(d_edges);
        cudaFree(d_envs_ptr);
        auto cleanup_ns = get_elapsed_nanoseconds(cleanup_start_time);
        std::cout << "Cleanup time: " << cleanup_ns << " ns" << std::endl;
    }

    template void batch_cc<typename ppln::robots::Panda>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Panda::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
    // template void batch_cc<typename ppln::robots::Fetch>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Fetch::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
    // template void batch_cc<typename ppln::robots::Baxter>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Baxter::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
} // namespace batch_cc