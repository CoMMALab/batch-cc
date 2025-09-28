#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/graph/graph_traits.hpp>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/Planners.hh"
#include "src/pRRTC_settings.hh"
#include "src/batch_cc.hh"
#include "src/random/halton.hh"

#include <vamp/planning/validate.hh>
#include <vamp/collision/environment.hh>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vamp/collision/factory.hh>
#include <vamp/robots/baxter.hh>
#include <vamp/robots/fetch.hh>
#include <vamp/robots/panda.hh>

// using json = nlohmann::json;

using namespace ppln::collision;


/* code to read scene txt file and create environments */
#define MAX_WORLD_SAMPLES_EVAL (200)
#define MAX_WORLD_SAMPLES (50)

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

// struct MeshData {
//     std::vector<vamp::collision::Point> points;
//     float r_min;
//     float r_max;
//     float r_point;

//     MeshData(std::vector<vamp::collision::Point> p, float min, float max, float rp)
//         : points(p), r_min(min), r_max(max), r_point(rp) {}
// };

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

std::vector<Environment<float>> setup_gpu_environments(std::string filename) {
    std::vector<Environment<float>> environments(MAX_WORLD_SAMPLES_EVAL);
    std::vector<Sphere<float>> object_spheres; // spheres for each environment
    std::vector<Cuboid<float>> object_cuboids; // cuboids for each environment
    std::vector<Capsule<float>> object_capsules; // cylinders for each environment

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

        iss >> category >> obj_type;

        std::string obs_name;
        if (category.find("OBSTACLE") != std::string::npos) {
            obs_name = category;
            if (!obs_name.empty() && obs_name.back() == '_') {
                obs_name.pop_back();
            }
        }
        if(obj_type == "BOX") {
            float x, y, z, qx, qy, qz, qw, dim_x, dim_y, dim_z;
            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> dim_x >> dim_y >> dim_z;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CuboidData cuboid(x, y, z, angles[0], angles[1], angles[2], dim_x, dim_y, dim_z);

            if (category.find("OBJECT") != std::string::npos) {
                object_cuboids.push_back(factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
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
                object_spheres.push_back(factory::sphere::array(sphere.center, radius));
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
                object_capsules.push_back(factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cylinders_map[obs_name].push_back(cylinder);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cylinders.push_back(cylinder);
            }
        }
    }

    env_file.close();

    // primitives coming from obstacles and targets for each environment
    std::vector<std::vector<Sphere<float>>> obstacle_target_spheres(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<Cuboid<float>>> obstacle_target_cuboids(MAX_WORLD_SAMPLES_EVAL);
    std::vector<std::vector<Capsule<float>>> obstacle_target_capsules(MAX_WORLD_SAMPLES_EVAL);
    std::cout << "here1\n";
    for (const auto& cuboid_entry : obstacle_cuboids_map) {
        std::string obs_name = cuboid_entry.first;
        std::vector<CuboidData> cuboids = cuboid_entry.second;

        if(cuboids.size() == 0)
            continue;
        if(cuboids.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cuboids)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            obstacle_target_cuboids[i].push_back(factory::cuboid::array(cuboids[i].center, cuboids[i].euler_angles, cuboids[i].half_dimens));
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
            obstacle_target_spheres[i].push_back(factory::sphere::array(spheres[i].center, spheres[i].radius));
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
            obstacle_target_capsules[i].push_back(factory::cylinder::center::array(cylinders[i].center,
                            cylinders[i].euler_angles, cylinders[i].radius, cylinders[i].length));
    }

    assert(("Buzz! Evironment samples not matched (Target Cuboids)! " + std::to_string(target_cuboids.size()),
                                        ((target_cuboids.size() >= MAX_WORLD_SAMPLES) || target_cuboids.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Spheres)! " + std::to_string(target_spheres.size()),
                                        ((target_spheres.size() >= MAX_WORLD_SAMPLES) || target_spheres.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Cylinders)! " + std::to_string(target_cylinders.size()),
                                        ((target_cylinders.size() >= MAX_WORLD_SAMPLES) || target_cylinders.size() == 0)));
    for (size_t i = 0; i < target_cuboids.size(); ++i) {
        if(i >= environments.size())
            break;
        obstacle_target_cuboids[i].push_back(factory::cuboid::array(target_cuboids[i].center, target_cuboids[i].euler_angles, target_cuboids[i].half_dimens));
    }
    for (size_t i = 0; i < target_spheres.size(); ++i) {
        if(i >= environments.size())
            break;
        obstacle_target_spheres[i].push_back(factory::sphere::array(target_spheres[i].center, target_spheres[i].radius));
    }
    for (size_t i = 0; i < target_cylinders.size(); ++i) {
        if(i >= environments.size())
            break;
        obstacle_target_capsules[i].push_back(factory::cylinder::center::array(target_cylinders[i].center,
                            target_cylinders[i].euler_angles, target_cylinders[i].radius, target_cylinders[i].length));
    }

    // allocate memory for each array of primitves for each environment and copy the data over from the vectors
    for (std::size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
        std::size_t spheres_size = object_spheres.size() + obstacle_target_spheres[i].size();
        std::size_t cuboids_size = object_cuboids.size() + obstacle_target_cuboids[i].size();
        std::size_t capsules_size = object_capsules.size() + obstacle_target_capsules[i].size();
        if (spheres_size > 0) {
            environments[i].spheres = new Sphere<float>[spheres_size];
            std::copy(object_spheres.begin(), object_spheres.end(), environments[i].spheres);
            std::copy(obstacle_target_spheres[i].begin(), obstacle_target_spheres[i].end(), environments[i].spheres + object_spheres.size());
            environments[i].num_spheres = spheres_size;
        }
        if (cuboids_size > 0) {
            environments[i].cuboids = new Cuboid<float>[cuboids_size];
            std::copy(object_cuboids.begin(), object_cuboids.end(), environments[i].cuboids);
            std::copy(obstacle_target_cuboids[i].begin(), obstacle_target_cuboids[i].end(), environments[i].cuboids + object_cuboids.size());
            environments[i].num_cuboids = cuboids_size;
        }
        if (capsules_size > 0) {
            environments[i].capsules = new Capsule<float>[capsules_size];
            std::copy(object_capsules.begin(), object_capsules.end(), environments[i].capsules);
            std::copy(obstacle_target_capsules[i].begin(), obstacle_target_capsules[i].end(), environments[i].capsules + object_capsules.size());
            environments[i].num_capsules = capsules_size;
        }
    }

    // std::cout << "###################################################################\n";
    // std::cout << "Finish parsing the scene file and populating the vamp environment!" << std::endl;
    // std::cout << env_static.z_aligned_cuboids.size() + env_static.cuboids.size() +
    //         env_static.spheres.size() + env_static.cylinders.size() << " | ";
    // for (size_t i = 0; i < environments.size(); ++i) {
    //     std::cout << environments[i].z_aligned_cuboids.size() + environments[i].cuboids.size() +
    //                  environments[i].spheres.size() + environments[i].cylinders.size() << " | ";
    // }
    // std::cout << std::endl;
    // std::cout << "###################################################################\n";

    return environments;
}
/* end code to read scene txt file and create environments */


/* code to setup vamp environment*/
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
static EnvironmentInput env_static;
static EnvironmentVector env_static_vec;
static std::array<EnvironmentInput, MAX_WORLD_SAMPLES_EVAL> environments_obs;
static std::array<EnvironmentVector, MAX_WORLD_SAMPLES_EVAL> env_obs_vectors;
static std::array<EnvironmentInput, MAX_WORLD_SAMPLES_EVAL> environments_tgt;
static std::array<EnvironmentVector, MAX_WORLD_SAMPLES_EVAL> env_tgt_vectors;


template<typename VampRobot>
std::pair<std::vector<EnvironmentInput>, std::vector<EnvironmentVector>> setup_vamp_environments(std::string filename) {
    std::vector<EnvironmentInput> environments(MAX_WORLD_SAMPLES_EVAL);
    std::vector<EnvironmentVector> env_vectors(MAX_WORLD_SAMPLES_EVAL);
    /* the object are static objects */
    std::vector<CuboidData> object_cuboids;
    std::vector<SphereData> object_spheres;
    std::vector<CylinderData> object_cylinders;
    // std::vector<MeshData> object_meshes;

    /* Define maps for obstacles, as we may have more than one obstacle */
    std::map<std::string, std::vector<CuboidData>> obstacle_cuboids_map;
    std::map<std::string, std::vector<SphereData>> obstacle_spheres_map;
    std::map<std::string, std::vector<CylinderData>> obstacle_cylinders_map;
    // std::map<std::string, std::vector<MeshData>> obstacle_meshes_map;

    std::vector<CuboidData> target_cuboids;
    std::vector<SphereData> target_spheres;
    std::vector<CylinderData> target_cylinders;
    // std::vector<MeshData> target_meshes;

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
                    env.cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
                }
                for (auto& env : environments_obs) {
                    env.cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
                }
                env_static.cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
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

    // for (const auto& mesh_entry : obstacle_meshes_map) {
    //     std::string obs_name = mesh_entry.first;
    //     std::vector<MeshData> meshes = mesh_entry.second;

    //     if(meshes.size() == 0)
    //         continue;
    //     if(meshes.size() < MAX_WORLD_SAMPLES_EVAL) {
    //         std::cerr << "Buzz! Evironment samples not matched (Obstacle Meshes)!" << std::endl;
    //         exit(-1);
    //     }
    //     for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i) {
    //         environments[i].pointclouds.emplace_back(meshes[i].points, meshes[i].r_min, meshes[i].r_max, meshes[i].r_point);
    //         environments_obs[i].pointclouds.emplace_back(meshes[i].points, meshes[i].r_min, meshes[i].r_max, meshes[i].r_point);
    //     }
    // }

    assert(("Buzz! Evironment samples not matched (Target Cuboids)! " + std::to_string(target_cuboids.size()),
                                        ((target_cuboids.size() >= MAX_WORLD_SAMPLES_EVAL) || target_cuboids.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Spheres)! " + std::to_string(target_spheres.size()),
                                        ((target_spheres.size() >= MAX_WORLD_SAMPLES_EVAL) || target_spheres.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Cylinders)! " + std::to_string(target_cylinders.size()),
                                        ((target_cylinders.size() >= MAX_WORLD_SAMPLES_EVAL) || target_cylinders.size() == 0)));
    // assert(("Buzz! Evironment samples not matched (Target Meshes)! " + std::to_string(target_meshes.size()),
    //                                     ((target_meshes.size() >= MAX_WORLD_SAMPLES_EVAL) || target_meshes.size() == 0)));

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
    // for (size_t i = 0; i < target_meshes.size(); ++i) {
    //     if(i >= environments.size())
    //         break;
    //     environments[i].pointclouds.emplace_back(target_meshes[i].points, target_meshes[i].r_min, target_meshes[i].r_max, target_meshes[i].r_point);
    //     environments_tgt[i].pointclouds.emplace_back(target_meshes[i].points, target_meshes[i].r_min, target_meshes[i].r_max, target_meshes[i].r_point);
    // }

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
    // std::cout << "MESH:" << env_static.pointclouds.size() << " | ";
    // for (size_t i = 0; i < environments.size(); ++i) {
    //     std::cout << environments[i].pointclouds.size() << " | ";
    // }
    // std::cout << std::endl;
    // std::cout << "###################################################################\n";
    return {environments, env_vectors};
}
/* end of code to setup vamp environment */

/* code to read graph */
struct VertexProps
{
    std::string         states_raw;   // exact string from the .dot
    std::vector<double> states;       // parsed numbers
};


using Graph =
    boost::adjacency_list<boost::vecS,      // edges stored in vectors
                          boost::vecS,      // vertices stored in vectors
                          boost::undirectedS, // “graph G { … }” = undirected
                          VertexProps>;     // bundled vertex properties

void parse_states(VertexProps& vp)
{
    std::istringstream iss(vp.states_raw);
    double value;
    while (iss >> value) vp.states.push_back(value);
}

Graph read_graph_from_file(const std::string& file_path)
{
    std::ifstream in(file_path);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    Graph g;

    /* Tell Boost which attributes we care about.
       The special token  boost::ignore_other_properties  lets the loader
       silently skip anything else it encounters (e.g. edge weights).     */
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("states", boost::get(&VertexProps::states_raw, g));

    /* ------------------------------------------------------------------ */
    /*                       Read the GraphViz file                       */
    /* ------------------------------------------------------------------ */
    if (!read_graphviz(in, g, dp, "node_id")) {   // "node_id" → use vertex names
        throw std::runtime_error("Error reading GraphViz file: " + file_path);
    }

    /* Convert every raw string into a vector<double> */
    for (auto v : boost::make_iterator_range(vertices(g)))
        parse_states(g[v]);

    return g;
}
/* end of code to read graph*/


template<typename VampRobot>
void vamp_batch_cc(std::vector<EnvironmentVector>& vamp_envs, std::vector<std::array<typename VampRobot::Configuration, 2>>& vamp_edges_vec, int resolution, std::vector<bool>& results) {
    std::size_t num_edges = vamp_edges_vec.size();
    std::size_t num_envs = vamp_envs.size();
    std::cout << "resolution: " << VampRobot::resolution << "\n";
    for (int i = 0; i < num_edges; i++) {
        auto& edge = vamp_edges_vec[i];
        for (int j = 0; j < num_envs; j++) {
            // printf("Environment %d, num_spheres: %d, num_cuboids: %d, num_cylinders: %d\n", j, vamp_envs[j].spheres.size(), vamp_envs[j].cuboids.size(), vamp_envs[j].cylinders.size());
            // if (j == 20) break;
            auto& env = vamp_envs[j];
            auto& start = edge[0];
            auto& end = edge[1];
            results[i * num_envs + j] = not vamp::planning::validate_motion<VampRobot, rake, VampRobot::resolution>(start, end, env);
        }
        // if (i == 20) break;
    }
}

// Helper function to convert VAMP environment to Python dict format
void print_environment_as_python_dict(const EnvironmentInput& env, int env_index = 0) {
    std::cout << "# Environment " << env_index << " as Python dict format:\n";
    std::cout << "problem = {\n";
    std::cout << "    'problem': 'general',\n";
    
    // Print spheres
    std::cout << "    'sphere': [\n";
    for (size_t i = 0; i < env.spheres.size(); ++i) {
        const auto& sphere = env.spheres[i];
        std::cout << "        {\n";
        std::cout << "            'position': [" << sphere.x << ", " << sphere.y << ", " << sphere.z << "],\n";
        std::cout << "            'radius': " << sphere.r << ",\n";
        std::cout << "            'name': '" << sphere.name << "'\n";
        std::cout << "        }";
        if (i < env.spheres.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "    ],\n";
    
    // Print cylinders
    std::cout << "    'cylinder': [\n";
    for (size_t i = 0; i < env.cylinders.size(); ++i) {
        const auto& capsule = env.cylinders[i];
        
        // Calculate position (midpoint of cylinder)
        float x_center = capsule.x1 + capsule.xv * 0.5f;
        float y_center = capsule.y1 + capsule.yv * 0.5f;
        float z_center = capsule.z1 + capsule.zv * 0.5f;
        
        // Calculate orientation quaternion from cylinder vector
        // The cylinder vector represents the direction from x1,y1,z1 to x2,y2,z2
        float length = std::sqrt(capsule.xv * capsule.xv + capsule.yv * capsule.yv + capsule.zv * capsule.zv);
        Eigen::Vector3f direction(capsule.xv / length, capsule.yv / length, capsule.zv / length);
        
        // Create rotation from default z-axis (0,0,1) to the cylinder direction
        Eigen::Vector3f default_axis(0, 0, 1);
        Eigen::Vector3f axis = default_axis.cross(direction);
        
        // Handle case where direction is parallel to default axis
        if (axis.norm() < 1e-6) {
            if (direction.dot(default_axis) > 0) {
                // Same direction - no rotation needed
                axis = Eigen::Vector3f(0, 0, 1);
            } else {
                // Opposite direction - 180 degree rotation around any perpendicular axis
                axis = Eigen::Vector3f(1, 0, 0);
            }
        } else {
            axis.normalize();
        }
        
        float angle = std::acos(std::clamp(default_axis.dot(direction), -1.0f, 1.0f));
        Eigen::AngleAxisf rotation(angle, axis);
        Eigen::Quaternionf quat(rotation);
        
        // Convert quaternion to Euler angles
        auto euler_angles = quaternion_to_euler(quat.x(), quat.y(), quat.z(), quat.w());
        
        std::cout << "        {\n";
        std::cout << "            'position': [" << x_center << ", " << y_center << ", " << z_center << "],\n";
        std::cout << "            'orientation_quat_xyzw': [" << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << "],\n";
        std::cout << "            'orientation_euler_xyz': [" << euler_angles[0] << ", " << euler_angles[1] << ", " << euler_angles[2] << "],\n";
        std::cout << "            'radius': " << capsule.r << ",\n";
        std::cout << "            'length': " << length << ",\n";
        std::cout << "            'name': '" << capsule.name << "'\n";
        std::cout << "        }";
        if (i < env.cylinders.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "    ],\n";
    
    // Print cuboids
    std::cout << "    'box': [\n";
    for (size_t i = 0; i < env.cuboids.size(); ++i) {
        const auto& cuboid = env.cuboids[i];
        
        // Calculate half extents
        float half_x = cuboid.axis_1_r;
        float half_y = cuboid.axis_2_r;
        float half_z = cuboid.axis_3_r;
        
        // Convert axis representation to quaternion
        // The three axes define the orientation of the cuboid
        Eigen::Matrix3f rotation_matrix;
        rotation_matrix.col(0) = Eigen::Vector3f(cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z);
        rotation_matrix.col(1) = Eigen::Vector3f(cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z);
        rotation_matrix.col(2) = Eigen::Vector3f(cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z);
        
        Eigen::Quaternionf quat(rotation_matrix);
        
        // Convert quaternion to Euler angles
        auto euler_angles = quaternion_to_euler(quat.x(), quat.y(), quat.z(), quat.w());
        
        std::cout << "        {\n";
        std::cout << "            'position': [" << cuboid.x << ", " << cuboid.y << ", " << cuboid.z << "],\n";
        std::cout << "            'orientation_quat_xyzw': [" << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << "],\n";
        std::cout << "            'orientation_euler_xyz': [" << euler_angles[0] << ", " << euler_angles[1] << ", " << euler_angles[2] << "],\n";
        std::cout << "            'half_extents': [" << half_x << ", " << half_y << ", " << half_z << "],\n";
        std::cout << "            'name': '" << cuboid.name << "'\n";
        std::cout << "        }";
        if (i < env.cuboids.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "    ]\n";
    
    std::cout << "}\n";
    std::cout << "\n";
}

template<typename Robot, typename VampRobot>
void run_test(std::string graph_file_path, std::string scene_file_path, int resolution, std::string robot_name) {
    // std::cout << "Running test for robot: " << robot_name << "\n";
    // std::cout << "Creating environments from scene file: " << scene_file_path << "\n";
    std::vector<Environment<float>> h_envs = setup_gpu_environments(scene_file_path);
    auto [vamp_envs_input, vamp_envs] = setup_vamp_environments<VampRobot>(scene_file_path);
    // std::cout << "Number of environments: " << h_envs.size() << "\n";
    
    // Print the first environment as Python dict format for visualization
    // if (!h_envs.empty()) {
    //     print_environment_as_python_dict(environments_obs[0], 0);
    // }
    // std::cout << "Creating graph from file: " << graph_file_path << "\n";
    Graph g = read_graph_from_file(graph_file_path);
    // std::cout << "Number of vertices: " << boost::num_vertices(g) << "\n";
    // std::cout << "Number of edges: " << boost::num_edges(g) << "\n";
    std::vector<std::array<typename Robot::Configuration, 2>> edges_vec;
    std::vector<std::array<typename VampRobot::Configuration, 2>> vamp_edges_vec;
    // add all edges from graph to edges vector
    for (auto e : boost::make_iterator_range(edges(g))) {
        auto src = source(e, g);
        auto tgt = target(e, g);

        // print src and tgt
        // for (int j = 0; j < Robot::dimension; j++) {
        //     std::cout << g[src].states[j] << " ";
        // }
        // std::cout << " -> ";
        // for (int j = 0; j < Robot::dimension; j++) {
        //     std::cout << g[tgt].states[j] << " ";
        // }
        // std::cout << "\n";

        std::array<typename Robot::Configuration, 2> edge;
        std::array<typename VampRobot::Configuration, 2> vamp_edge;
        typename Robot::Configuration start, end;
        for (int i = 0; i < Robot::dimension; i++) {
            
            edge[0][i] = g[src].states[i];
            edge[1][i] = g[tgt].states[i];
            start[i] = g[src].states[i];
            end[i] = g[tgt].states[i];
        }

        typename VampRobot::Configuration start_vamp(start);
        typename VampRobot::Configuration end_vamp(end);
        vamp_edge[0] = start_vamp;
        vamp_edge[1] = end_vamp;
        edges_vec.push_back(edge);
        vamp_edges_vec.push_back(vamp_edge);
    }

    std::size_t num_edges = edges_vec.size();
    std::size_t num_envs = h_envs.size();
    std::cout << "Number of edges: " << num_edges << "\n";
    std::cout << "Number of environments: " << num_envs << "\n";
    
    std::vector<bool> results(num_edges * num_envs, false);
    std::vector<bool> vamp_results(num_edges * num_envs, false);
    auto start = std::chrono::high_resolution_clock::now();
    batch_cc::batch_cc<Robot>(h_envs, edges_vec, resolution, results);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
    std::cout << "Batch collision checking took: " << ns.count() / 1'000'000'000.0 << " s\n";

    vamp_batch_cc<VampRobot>(vamp_envs, vamp_edges_vec, resolution, vamp_results);

    for (int i = 0; i < num_edges; i++) {
        for (int j = 0; j < num_envs; j++) {
            bool gpu_result = results[i * num_envs + j];
            bool vamp_result = vamp_results[i * num_envs + j];
            if (gpu_result != vamp_result) {
                printf("Discrepancy at env %d, edge %d: gpu-{%d} vamp-{%d}\n", j, i, gpu_result, vamp_result);
                for (int k = 0; k < Robot::dimension; k++) {
                    std::cout << edges_vec[i][0][k] << " ";
                }
                std::cout << " -> ";
                for (int k = 0; k < Robot::dimension; k++) {
                    std::cout << edges_vec[i][1][k] << " ";
                }
                std::cout << "\n";
                print_environment_as_python_dict(vamp_envs_input[j], j);
                return;
            }
            // printf("%d", gpu_result);
        }
        // printf("\n");
    }
    std::cout << "All correct!\n";
    
}



int main(int argc, char* argv[]) {
    std::string robot_name = "panda";
    std::string graph_file_path = "graph.dot";
    std::string scene_file_path = "scene.txt";
    int resolution = 32;
    if (argc == 4) {
        robot_name = argv[1];
        graph_file_path = argv[2];
        scene_file_path = argv[3];
    }
    else {
        std::cout << "Usage: ./batch_cc <robot_name> <graph.dot> <scene.txt>\n";
        return 1;
    }
    if (robot_name == "panda") {
        run_test<robots::Panda, vamp::robots::Panda>(graph_file_path, scene_file_path, resolution, robot_name);
    }
    else if (robot_name == "fetch") {
        run_test<robots::Fetch, vamp::robots::Fetch>(graph_file_path, scene_file_path, resolution, robot_name);
    }
    else {
        std::cout << "Unknown robot name: " << robot_name << "\n";
        return -1;
    }
    return 0;
}

