#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <algorithm>

#include "vamp/collision/factory.hh"
#include "vamp/collision/math.hh"
#include "vamp/collision/filter.hh"
#include "vamp/planning/validate.hh"
#include "vamp/robots/panda.hh"

#include "vamp_gpu/collision/environment.hh"
#include "vamp_gpu/collision/factory.hh"
#include "vamp_gpu/Planners.hh"
#include "vamp_gpu/pRRTC_settings.hh"
#include "vamp_gpu/batch_cc.hh"
#include "vamp_gpu/random/halton.hh"

using VampRobot = vamp::robots::Panda;
using VampRobotG = ppln::robots::Panda;

static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
static std::array<EnvironmentInput, MAX_WORLD_SAMPLES_EVAL> environments;
static std::array<EnvironmentVector, MAX_WORLD_SAMPLES_EVAL> env_vectors;
static EnvironmentInput env_static;
static EnvironmentVector env_static_vec;

/* below four vectors are for the rutger's baseline algorithm */
static std::array<EnvironmentInput, MAX_WORLD_SAMPLES_EVAL> environments_obs;
static std::array<EnvironmentVector, MAX_WORLD_SAMPLES_EVAL> env_obs_vectors;
static std::array<EnvironmentInput, MAX_WORLD_SAMPLES_EVAL> environments_tgt;
static std::array<EnvironmentVector, MAX_WORLD_SAMPLES_EVAL> env_tgt_vectors;

static std::vector<ppln::collision::Environment<float>> env_vectors_gpu(MAX_WORLD_SAMPLES_EVAL);
static std::vector<ppln::collision::Environment<float>> env_vectors_obs_gpu(MAX_WORLD_SAMPLES_EVAL);
static std::vector<ppln::collision::Environment<float>> env_vectors_tgt_gpu(MAX_WORLD_SAMPLES_EVAL);
static std::vector<ppln::collision::Environment<float>> env_static_vec_gpu(1);

#define BOX (1)
#define SPHERE (2)
#define CYLINDER (3)
#define CONE (4)
#define PRISM (5)

#define POINT_RADIUS                    0.0025
#define POINTCLOUDS_FILTER_RADIUS       0.004
#define POINTCLOUDS_PANDA_MAX_RANGE     1.19


struct CuboidData {
    const std::array<float, 3> center;
    const std::array<float, 3> euler_angles;
    const std::array<float, 3> half_dimens;

    CuboidData(float cx, float cy, float cz, float r, float p, float y, float dimX, float dimY, float dimZ)
        : center{cx, cy, cz},
          euler_angles{r, p, y},
          half_dimens{dimX / 2, dimY / 2, dimZ / 2} {}
};

struct SphereData {
    std::array<float, 3> center;
    float radius;

    SphereData(float cx, float cy, float cz, float r)
        : center{cx, cy, cz},
          radius(r) {}
};

struct CylinderData {
    std::array<float, 3> center;
    std::array<float, 3> euler_angles;
    float radius;
    float length;

    CylinderData(float cx, float cy, float cz, float r, float p, float y, float rad, float len)
        : center{cx, cy, cz},
          euler_angles{r, p, y},
          radius(rad), length(len) {}
};

struct MeshData {
    std::vector<vamp::collision::Point> points;
    float r_min;
    float r_max;
    float r_point;

    MeshData(std::vector<vamp::collision::Point> p, float min, float max, float rp)
        : points(p), r_min(min), r_max(max), r_point(rp) {}
};

const std::vector<std::array<float, 3>> problem_ = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    {0, -0.55, 0.25},
    {0.35, -0.35, 0.25},
    {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},
    {-0.35, 0.35, 0.8},
    {-0.55, 0, 0.8},
    {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},
    {0.35, -0.35, 0.8},
    };

inline std::array<float, 3> quaternion_to_euler(float x, float y, float z, float w) {
    std::array<float, 3> angles; // angles[0]: roll, angles[1]: pitch, angles[2]: yaw

    // Roll (x-axis rotation)
    float t0 = 2.0f * (w * x + y * z);
    float t1 = 1.0f - 2.0f * (x * x + y * y);
    angles[0] = std::atan2(t0, t1);

    // Pitch (y-axis rotation)
    float t2 = 2.0f * (w * y - z * x);
    // Clamp t2 to be in the range [-1, 1] to avoid errors due to floating-point inaccuracies.
    t2 = std::max(-1.0f, std::min(1.0f, t2));
    angles[1] = std::asin(t2);

    // Yaw (z-axis rotation)
    float t3 = 2.0f * (w * z + x * y);
    float t4 = 1.0f - 2.0f * (y * y + z * z);
    angles[2] = std::atan2(t3, t4);

    return angles;
}


inline std::vector<vamp::collision::Point> get_mesh_points(std::string path)
{
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Cannot open OBJ " + path);

    std::vector<vamp::collision::Point> vertices;
    std::vector<vamp::collision::Point> filtered_vertices;

    static size_t n = 0, last_n = 0;

    std::string line;
    while(std::getline(in, line)) {
        if(line.empty() || line[0] == '#') continue;       // comment
        std::istringstream iss(line);
        std::string tag; iss >> tag;

        // vertex position
        if(tag == "v") {
            float x,y,z;
            iss >> x >> y >> z;
            vamp::collision::Point p{x, y, z};
            vertices.emplace_back(p);
        }
    }

    filtered_vertices = vamp::collision::filter_pointcloud(
        vertices,                                               /* original points */
        POINTCLOUDS_FILTER_RADIUS,                              /* minimal distance */
        POINTCLOUDS_PANDA_MAX_RANGE,                            /* maximum distance */
        vamp::collision::Point{0.0, 0.0, 0.0},                  /* origin */
        vamp::collision::Point{-POINTCLOUDS_PANDA_MAX_RANGE,
                               -POINTCLOUDS_PANDA_MAX_RANGE,
                               -POINTCLOUDS_PANDA_MAX_RANGE},    /* AABB minimum vertex */
        vamp::collision::Point{ POINTCLOUDS_PANDA_MAX_RANGE,
                                POINTCLOUDS_PANDA_MAX_RANGE,
                                POINTCLOUDS_PANDA_MAX_RANGE},    /* AABB maximum vertex */
        true                                                     /* cull pointcloud around robot by maximum distance */
    );

    n = vertices.size();
    if(last_n != n) {
        std::cout << "read " << vertices.size() << " points from mesh file, ";
        std::cout << filtered_vertices.size() << " remaining after filtering." << std::endl;
    }
    last_n = n;

    return filtered_vertices;
}

inline void setup_vamp_environments(std::string filename)
{
    /* the object are static objects */
    std::vector<CuboidData> object_cuboids;
    std::vector<SphereData> object_spheres;
    std::vector<CylinderData> object_cylinders;
    std::vector<MeshData> object_meshes;

    /* Define maps for obstacles, as we may have more than one obstacle */
    std::map<std::string, std::vector<CuboidData>> obstacle_cuboids_map;
    std::map<std::string, std::vector<SphereData>> obstacle_spheres_map;
    std::map<std::string, std::vector<CylinderData>> obstacle_cylinders_map;
    std::map<std::string, std::vector<MeshData>> obstacle_meshes_map;

    std::vector<CuboidData> target_cuboids;
    std::vector<SphereData> target_spheres;
    std::vector<CylinderData> target_cylinders;
    std::vector<MeshData> target_meshes;

    std::ifstream env_file(filename);
    if (!env_file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(-1);
    }

    std::string line;

    while (std::getline(env_file, line)) {
        std::istringstream iss(line);
        std::string category, obj_type;

        iss >> category;

        if((category == "TARGET") ||
           ((category.find("OBSTACLE") != std::string::npos) && (category.back() != '_'))) {
            /* skip the mean of target and obstacles */
            continue;
        }

        std::string obs_name;
        if ((category.find("OBSTACLE") != std::string::npos) && (category.back() == '_')) {
            obs_name = category.substr(0, category.size() - 1);
        }

        iss >> obj_type;
        if(obj_type == "BOX") {
            float x, y, z, qx, qy, qz, qw, dim_x, dim_y, dim_z;
            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> dim_x >> dim_y >> dim_z;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CuboidData cuboid(x, y, z, angles[0], angles[1], angles[2], dim_x, dim_y, dim_z);

            if (category.find("OBJECT") != std::string::npos) {
                object_cuboids.push_back(cuboid);
                for (auto& env : environments) {
                    env.z_aligned_cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
                }
                for (auto& env : environments_obs) {
                    env.z_aligned_cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
                }
                env_static.z_aligned_cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cuboids_map[obs_name].push_back(cuboid);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cuboids.push_back(cuboid);
            }
        } else if(obj_type == "SPHERE") {
            float x, y, z, radius;

            iss >> x >> y >> z >> radius;
            const SphereData sphere(x, y, z, radius);

            if (category.find("OBJECT") != std::string::npos) {
                object_spheres.push_back(sphere);
                for (auto& env : environments) {
                    env.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere.center, radius));
                }
                for (auto& env : environments_obs) {
                    env.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere.center, radius));
                }
                env_static.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere.center, radius));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_spheres_map[obs_name].push_back(sphere);
            } else if (category.find("TARGET") != std::string::npos) {
                target_spheres.push_back(sphere);
            }
        } else if(obj_type == "CYLINDER") {
            float x, y, z, qx, qy, qz, qw, height, radius;

            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> height >> radius;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CylinderData cylinder(x, y, z, angles[0], angles[1], angles[2], radius, height);

            if (category.find("OBJECT") != std::string::npos) {
                object_cylinders.push_back(cylinder);
                for (auto& env : environments) {
                    env.cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
                }
                for (auto& env : environments_obs) {
                    env.cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
                }
                env_static.cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cylinders_map[obs_name].push_back(cylinder);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cylinders.push_back(cylinder);
            }
        } else if(obj_type == "MESH") {
            float x, y, z, qx, qy, qz, qw;
            std::string mesh_path;

            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> mesh_path;
            std::vector<vamp::collision::Point> p = get_mesh_points(mesh_path);

            Eigen::Quaternionf q(qw, qx, qy, qz);   // Eigen = (w,x,y,z)
            q.normalize();                            // protect against bad input
            const Eigen::Matrix3f R = q.toRotationMatrix();
            const Eigen::Vector3f t(x, y, z);

            for (vamp::collision::Point& v : p) {                    // modify the same vector
                Eigen::Vector3f loc(v[0], v[1], v[2]);
                Eigen::Vector3f w = R * loc + t;
                v = { w.x(), w.y(), w.z() };
            }

            const MeshData mesh(std::move(p), VampRobot::min_radius, VampRobot::max_radius, POINT_RADIUS);

            if (category.find("OBJECT") != std::string::npos) {
                object_meshes.push_back(mesh);
                for (auto& env : environments) {
                    env.pointclouds.emplace_back(mesh.points, mesh.r_min, mesh.r_max, mesh.r_point);
                }
                for (auto& env : environments_obs) {
                    env.pointclouds.emplace_back(mesh.points, mesh.r_min, mesh.r_max, mesh.r_point);
                }
                env_static.pointclouds.emplace_back(mesh.points, mesh.r_min, mesh.r_max, mesh.r_point);
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_meshes_map[obs_name].push_back(mesh);
            } else if (category.find("TARGET") != std::string::npos) {
                target_meshes.push_back(mesh);
            }
        }
    }

    env_static.sort();
    env_static_vec = EnvironmentVector(env_static);
    env_file.close();

    for (const auto& cuboid_entry : obstacle_cuboids_map) {
        std::string obs_name = cuboid_entry.first;
        std::vector<CuboidData> cuboids = cuboid_entry.second;

        if(cuboids.size() == 0)
            continue;
        if(cuboids.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cuboids)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
            environments[i].cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboids[i].center, cuboids[i].euler_angles, cuboids[i].half_dimens));
            environments[i].sort();
            environments_obs[i].cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboids[i].center, cuboids[i].euler_angles, cuboids[i].half_dimens));
            environments_obs[i].sort();
        }
    }

    for (const auto& sphere_entry : obstacle_spheres_map) {
        std::string obs_name = sphere_entry.first;
        std::vector<SphereData> spheres = sphere_entry.second;

        if(spheres.size() == 0)
            continue;
        if(spheres.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Sphere)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
            environments[i].spheres.emplace_back(vamp::collision::factory::sphere::array(spheres[i].center, spheres[i].radius));
            environments[i].sort();
            environments_obs[i].spheres.emplace_back(vamp::collision::factory::sphere::array(spheres[i].center, spheres[i].radius));
            environments_obs[i].sort();
        }
    }


    for (const auto& cylinder_entry : obstacle_cylinders_map) {
        std::string obs_name = cylinder_entry.first;
        std::vector<CylinderData> cylinders = cylinder_entry.second;

        if(cylinders.size() == 0)
            continue;
        if(cylinders.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cylinders)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
            environments[i].cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinders[i].center,
                            cylinders[i].euler_angles, cylinders[i].radius, cylinders[i].length));
            environments[i].sort();
            environments_obs[i].cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinders[i].center,
                            cylinders[i].euler_angles, cylinders[i].radius, cylinders[i].length));
            environments_obs[i].sort();
        }
    }

    for (const auto& mesh_entry : obstacle_meshes_map) {
        std::string obs_name = mesh_entry.first;
        std::vector<MeshData> meshes = mesh_entry.second;

        if(meshes.size() == 0)
            continue;
        if(meshes.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Meshes)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
            environments[i].pointclouds.emplace_back(meshes[i].points, meshes[i].r_min, meshes[i].r_max, meshes[i].r_point);
            environments_obs[i].pointclouds.emplace_back(meshes[i].points, meshes[i].r_min, meshes[i].r_max, meshes[i].r_point);
        }
    }

    assert(("Buzz! Evironment samples not matched (Target Cuboids)! " + std::to_string(target_cuboids.size()),
                                        ((target_cuboids.size() >= MAX_WORLD_SAMPLES_EVAL) || target_cuboids.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Spheres)! " + std::to_string(target_spheres.size()),
                                        ((target_spheres.size() >= MAX_WORLD_SAMPLES_EVAL) || target_spheres.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Cylinders)! " + std::to_string(target_cylinders.size()),
                                        ((target_cylinders.size() >= MAX_WORLD_SAMPLES_EVAL) || target_cylinders.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Meshes)! " + std::to_string(target_meshes.size()),
                                        ((target_meshes.size() >= MAX_WORLD_SAMPLES_EVAL) || target_meshes.size() == 0)));

    for (size_t i = 0; i < target_cuboids.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].cuboids.emplace_back(vamp::collision::factory::cuboid::array(target_cuboids[i].center, target_cuboids[i].euler_angles, target_cuboids[i].half_dimens));
        environments[i].sort();
        environments_tgt[i].cuboids.emplace_back(vamp::collision::factory::cuboid::array(target_cuboids[i].center, target_cuboids[i].euler_angles, target_cuboids[i].half_dimens));
        environments_tgt[i].sort();
    }
    for (size_t i = 0; i < target_spheres.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].spheres.emplace_back(vamp::collision::factory::sphere::array(target_spheres[i].center, target_spheres[i].radius));
        environments[i].sort();
        environments_tgt[i].spheres.emplace_back(vamp::collision::factory::sphere::array(target_spheres[i].center, target_spheres[i].radius));
        environments_tgt[i].sort();
    }
    for (size_t i = 0; i < target_cylinders.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(target_cylinders[i].center,
                            target_cylinders[i].euler_angles, target_cylinders[i].radius, target_cylinders[i].length));
        environments[i].sort();
        environments_tgt[i].cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(target_cylinders[i].center,
                            target_cylinders[i].euler_angles, target_cylinders[i].radius, target_cylinders[i].length));
        environments_tgt[i].sort();
    }
    for (size_t i = 0; i < target_meshes.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].pointclouds.emplace_back(target_meshes[i].points, target_meshes[i].r_min, target_meshes[i].r_max, target_meshes[i].r_point);
        environments_tgt[i].pointclouds.emplace_back(target_meshes[i].points, target_meshes[i].r_min, target_meshes[i].r_max, target_meshes[i].r_point);
    }

    for (size_t i = 0; i < environments.size(); ++i) {
        env_vectors[i] = EnvironmentVector(environments[i]);
        env_obs_vectors[i] = EnvironmentVector(environments_obs[i]);
        env_tgt_vectors[i] = EnvironmentVector(environments_tgt[i]);
    }

    std::cout << "###################################################################\n";
    std::cout << "Finish parsing the scene file and populating the vamp environment!" << std::endl;
    std::cout << "PRIM:" << env_static.z_aligned_cuboids.size() + env_static.cuboids.size() +
            env_static.spheres.size() + env_static.cylinders.size() << " | ";
    for (size_t i = 0; i < environments.size(); ++i) {
        std::cout << environments[i].z_aligned_cuboids.size() + environments[i].cuboids.size() +
                     environments[i].spheres.size() + environments[i].cylinders.size() << " | ";
    }
    std::cout << std::endl;
    std::cout << "MESH:" << env_static.pointclouds.size() << " | ";
    for (size_t i = 0; i < environments.size(); ++i) {
        std::cout << environments[i].pointclouds.size() << " | ";
    }
    std::cout << std::endl;
    std::cout << "###################################################################\n";
}


inline void setup_vamp_gpu_environments(std::string filename)
{
    // the last element is for static environment
    std::vector<ppln::collision::Sphere<float>> object_spheres; // spheres for each environment
    std::vector<ppln::collision::Cuboid<float>> object_cuboids; // cuboids for each environment
    std::vector<ppln::collision::Capsule<float>> object_capsules; // cylinders for each environment

    /* Define maps for obstacles, as we may have more than one obstacle */
    std::map<std::string, std::vector<CuboidData>> obstacle_cuboids_map;
    std::map<std::string, std::vector<SphereData>> obstacle_spheres_map;
    std::map<std::string, std::vector<CylinderData>> obstacle_cylinders_map;

    std::vector<CuboidData> target_cuboids;
    std::vector<SphereData> target_spheres;
    std::vector<CylinderData> target_cylinders;

    std::ifstream env_file(filename);
    if (!env_file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::exit(-1);
    }

    std::string line;

    while (std::getline(env_file, line)) {
        std::istringstream iss(line);
        std::string category, obj_type;

        iss >> category;
        if((category == "TARGET") ||
           ((category.find("OBSTACLE") != std::string::npos) && (category.back() != '_'))) {
            /* skip the mean of target and obstacles */
            continue;
        }

        std::string obs_name;
        if ((category.find("OBSTACLE") != std::string::npos) && (category.back() == '_')) {
            obs_name = category.substr(0, category.size() - 1);
        }

        iss >> obj_type;
        if(obj_type == "BOX") {
            float x, y, z, qx, qy, qz, qw, dim_x, dim_y, dim_z;
            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> dim_x >> dim_y >> dim_z;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CuboidData cuboid(x, y, z, angles[0], angles[1], angles[2], dim_x, dim_y, dim_z);

            if (category.find("OBJECT") != std::string::npos) {
                object_cuboids.push_back(ppln::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cuboids_map[obs_name].push_back(cuboid);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cuboids.push_back(cuboid);
            }
        } else if(obj_type == "SPHERE") {
            float x, y, z, radius;

            iss >> x >> y >> z >> radius;
            const SphereData sphere(x, y, z, radius);

            if (category.find("OBJECT") != std::string::npos) {
                object_spheres.push_back(ppln::collision::factory::sphere::array(sphere.center, radius));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_spheres_map[obs_name].push_back(sphere);
            } else if (category.find("TARGET") != std::string::npos) {
                target_spheres.push_back(sphere);
            }
        } else if(obj_type == "CYLINDER") {
            float x, y, z, qx, qy, qz, qw, height, radius;

            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> height >> radius;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CylinderData cylinder(x, y, z, angles[0], angles[1], angles[2], radius, height);

            if (category.find("OBJECT") != std::string::npos) {
                object_capsules.push_back(ppln::collision::factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cylinders_map[obs_name].push_back(cylinder);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cylinders.push_back(cylinder);
            }
        }
    }

    env_file.close();

    // primitives coming from obstacles and targets for each environment
    std::vector<std::vector<ppln::collision::Sphere<float>>> ppln_obstacle_spheres(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<ppln::collision::Cuboid<float>>> ppln_obstacle_cuboids(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<ppln::collision::Capsule<float>>> ppln_obstacle_capsules(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<ppln::collision::Sphere<float>>> ppln_target_spheres(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<ppln::collision::Cuboid<float>>> ppln_target_cuboids(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<ppln::collision::Capsule<float>>> ppln_target_capsules(MAX_WORLD_SAMPLES_EVAL);

    for (const auto& cuboid_entry : obstacle_cuboids_map) {
        std::string obs_name = cuboid_entry.first;
        std::vector<CuboidData> cuboids = cuboid_entry.second;

        if(cuboids.size() == 0)
            continue;
        if(cuboids.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cuboids)!" << std::endl;
            std::exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            ppln_obstacle_cuboids[i].push_back(ppln::collision::factory::cuboid::array(cuboids[i].center, cuboids[i].euler_angles, cuboids[i].half_dimens));
    }

    for (const auto& sphere_entry : obstacle_spheres_map) {
        std::string obs_name = sphere_entry.first;
        std::vector<SphereData> spheres = sphere_entry.second;

        if(spheres.size() == 0)
            continue;
        if(spheres.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Sphere)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            ppln_obstacle_spheres[i].push_back(ppln::collision::factory::sphere::array(spheres[i].center, spheres[i].radius));
    }


    for (const auto& cylinder_entry : obstacle_cylinders_map) {
        std::string obs_name = cylinder_entry.first;
        std::vector<CylinderData> cylinders = cylinder_entry.second;

        if(cylinders.size() == 0)
            continue;
        if(cylinders.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cylinders)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            ppln_obstacle_capsules[i].push_back(ppln::collision::factory::cylinder::center::array(cylinders[i].center,
                            cylinders[i].euler_angles, cylinders[i].radius, cylinders[i].length));
    }

    assert(("Buzz! Evironment samples not matched (Target Cuboids)! " + std::to_string(target_cuboids.size()),
                                        ((target_cuboids.size() >= MAX_WORLD_SAMPLES_EVAL) || target_cuboids.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Spheres)! " + std::to_string(target_spheres.size()),
                                        ((target_spheres.size() >= MAX_WORLD_SAMPLES_EVAL) || target_spheres.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Cylinders)! " + std::to_string(target_cylinders.size()),
                                        ((target_cylinders.size() >= MAX_WORLD_SAMPLES_EVAL) || target_cylinders.size() == 0)));
    for (size_t i = 0; i < target_cuboids.size(); ++i) {
        if(i >= env_vectors_gpu.size())
            break;
        ppln_target_cuboids[i].push_back(ppln::collision::factory::cuboid::array(target_cuboids[i].center, target_cuboids[i].euler_angles, target_cuboids[i].half_dimens));
    }
    for (size_t i = 0; i < target_spheres.size(); ++i) {
        if(i >= env_vectors_gpu.size())
            break;
        ppln_target_spheres[i].push_back(ppln::collision::factory::sphere::array(target_spheres[i].center, target_spheres[i].radius));
    }
    for (size_t i = 0; i < target_cylinders.size(); ++i) {
        if(i >= env_vectors_gpu.size())
            break;
        ppln_target_capsules[i].push_back(ppln::collision::factory::cylinder::center::array(target_cylinders[i].center,
                            target_cylinders[i].euler_angles, target_cylinders[i].radius, target_cylinders[i].length));
    }

    // allocate memory for each array of primitves for each environment and copy the data over from the vectors
    for (std::size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
        std::size_t spheres_size = object_spheres.size() + ppln_obstacle_spheres[i].size() + ppln_target_spheres[i].size();
        std::size_t cuboids_size = object_cuboids.size() + ppln_obstacle_cuboids[i].size() + ppln_target_cuboids[i].size();
        std::size_t capsules_size = object_capsules.size() + ppln_obstacle_capsules[i].size() + ppln_target_capsules[i].size();

        std::size_t obs_spheres_size = object_spheres.size() + ppln_obstacle_spheres[i].size();
        std::size_t obs_cuboids_size = object_cuboids.size() + ppln_obstacle_cuboids[i].size();
        std::size_t obs_capsules_size = object_capsules.size() + ppln_obstacle_capsules[i].size();

        std::size_t tgt_spheres_size = ppln_target_spheres[i].size();
        std::size_t tgt_cuboids_size = ppln_target_cuboids[i].size();
        std::size_t tgt_capsules_size = ppln_target_capsules[i].size();
        std::cerr << spheres_size << " " << cuboids_size << " " << capsules_size << " "
                  << obs_spheres_size << " " << obs_cuboids_size << " " << obs_capsules_size << " "
                  << tgt_spheres_size << " " << tgt_cuboids_size << " " << tgt_capsules_size
                  << std::endl;

        if (spheres_size > 0) {
            env_vectors_gpu[i].spheres = new ppln::collision::Sphere<float>[spheres_size];
            std::copy(object_spheres.begin(), object_spheres.end(), env_vectors_gpu[i].spheres);
            std::copy(ppln_obstacle_spheres[i].begin(), ppln_obstacle_spheres[i].end(), env_vectors_gpu[i].spheres + object_spheres.size());
            std::copy(ppln_target_spheres[i].begin(), ppln_target_spheres[i].end(), env_vectors_gpu[i].spheres + obs_spheres_size);
            env_vectors_gpu[i].num_spheres = spheres_size;

            env_vectors_obs_gpu[i].spheres = new ppln::collision::Sphere<float>[obs_spheres_size];
            std::copy(object_spheres.begin(), object_spheres.end(), env_vectors_obs_gpu[i].spheres);
            std::copy(ppln_obstacle_spheres[i].begin(), ppln_obstacle_spheres[i].end(), env_vectors_obs_gpu[i].spheres + object_spheres.size());
            env_vectors_obs_gpu[i].num_spheres = obs_spheres_size;

            env_vectors_tgt_gpu[i].spheres = new ppln::collision::Sphere<float>[tgt_spheres_size];
            std::copy(ppln_target_spheres[i].begin(), ppln_target_spheres[i].end(), env_vectors_tgt_gpu[i].spheres);
            env_vectors_tgt_gpu[i].num_spheres = tgt_spheres_size;
        }
        if (cuboids_size > 0) {
            env_vectors_gpu[i].cuboids = new ppln::collision::Cuboid<float>[cuboids_size];
            std::copy(object_cuboids.begin(), object_cuboids.end(), env_vectors_gpu[i].cuboids);
            std::copy(ppln_obstacle_cuboids[i].begin(), ppln_obstacle_cuboids[i].end(), env_vectors_gpu[i].cuboids + object_cuboids.size());
            std::copy(ppln_target_cuboids[i].begin(), ppln_target_cuboids[i].end(), env_vectors_gpu[i].cuboids + obs_cuboids_size);
            env_vectors_gpu[i].num_cuboids = cuboids_size;

            env_vectors_obs_gpu[i].cuboids = new ppln::collision::Cuboid<float>[obs_cuboids_size];
            std::copy(object_cuboids.begin(), object_cuboids.end(), env_vectors_obs_gpu[i].cuboids);
            std::copy(ppln_obstacle_cuboids[i].begin(), ppln_obstacle_cuboids[i].end(), env_vectors_obs_gpu[i].cuboids + object_cuboids.size());
            env_vectors_obs_gpu[i].num_cuboids = obs_cuboids_size;

            env_vectors_tgt_gpu[i].cuboids = new ppln::collision::Cuboid<float>[tgt_cuboids_size];
            std::copy(ppln_target_cuboids[i].begin(), ppln_target_cuboids[i].end(), env_vectors_tgt_gpu[i].cuboids);
            env_vectors_tgt_gpu[i].num_cuboids = tgt_cuboids_size;
        }
        if (capsules_size > 0) {
            env_vectors_gpu[i].capsules = new ppln::collision::Capsule<float>[capsules_size];
            std::copy(object_capsules.begin(), object_capsules.end(), env_vectors_gpu[i].capsules);
            std::copy(ppln_obstacle_capsules[i].begin(), ppln_obstacle_capsules[i].end(), env_vectors_gpu[i].capsules + object_capsules.size());
            std::copy(ppln_target_capsules[i].begin(), ppln_target_capsules[i].end(), env_vectors_gpu[i].capsules + obs_capsules_size);
            env_vectors_gpu[i].num_capsules = capsules_size;

            env_vectors_obs_gpu[i].capsules = new ppln::collision::Capsule<float>[obs_capsules_size];
            std::copy(object_capsules.begin(), object_capsules.end(), env_vectors_obs_gpu[i].capsules);
            std::copy(ppln_obstacle_capsules[i].begin(), ppln_obstacle_capsules[i].end(), env_vectors_obs_gpu[i].capsules + object_capsules.size());
            env_vectors_obs_gpu[i].num_capsules = obs_capsules_size;

            env_vectors_tgt_gpu[i].capsules = new ppln::collision::Capsule<float>[tgt_capsules_size];
            std::copy(ppln_target_capsules[i].begin(), ppln_target_capsules[i].end(), env_vectors_tgt_gpu[i].capsules);
            env_vectors_tgt_gpu[i].num_capsules = tgt_capsules_size;
        }
    }

    std::size_t spheres_size = object_spheres.size();
    std::size_t cuboids_size = object_cuboids.size();
    std::size_t capsules_size = object_capsules.size();
    if (spheres_size > 0) {
        env_static_vec_gpu[0].spheres = new ppln::collision::Sphere<float>[spheres_size];
        std::copy(object_spheres.begin(), object_spheres.end(), env_static_vec_gpu[0].spheres);
        env_static_vec_gpu[0].num_spheres = spheres_size;
    }
    if (cuboids_size > 0) {
        env_static_vec_gpu[0].cuboids = new ppln::collision::Cuboid<float>[cuboids_size];
        std::copy(object_cuboids.begin(), object_cuboids.end(), env_static_vec_gpu[0].cuboids);
        env_static_vec_gpu[0].num_cuboids = cuboids_size;
    }
    if (capsules_size > 0) {
        env_static_vec_gpu[0].capsules = new ppln::collision::Capsule<float>[capsules_size];
        std::copy(object_capsules.begin(), object_capsules.end(), env_static_vec_gpu[0].capsules);
        env_static_vec_gpu[0].num_capsules = capsules_size;
    }

    std::cout << "###################################################################\n";
    std::cout << "Finish parsing the scene file and populating the GPU environment!" << std::endl;
    for (size_t i = 0; i < env_static_vec_gpu.size(); ++i) {
        std::cout << env_static_vec_gpu[i].num_capsules + env_static_vec_gpu[i].num_cuboids +
            env_static_vec_gpu[i].num_spheres << " | ";
    }
    for (size_t i = 0; i < env_vectors_gpu.size(); ++i) {
        std::cout << env_vectors_gpu[i].num_capsules + env_vectors_gpu[i].num_cuboids +
            env_vectors_gpu[i].num_spheres << " | ";
    }
    std::cout << "\n";
    std::cout << "###################################################################\n";
}

