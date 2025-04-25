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

std::vector<Environment<float>> setup_environments(std::string filename) {

    std::vector<Environment<float>> environments(MAX_WORLD_SAMPLES_EVAL);
    std::vector<Sphere<float>> object_spheres(MAX_WORLD_SAMPLES_EVAL); // spheres for each environment
    std::vector<Cuboid<float>> object_cuboids(MAX_WORLD_SAMPLES_EVAL); // cuboids for each environment
    std::vector<Capsule<float>> object_capsules(MAX_WORLD_SAMPLES_EVAL); // cylinders for each environment

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

template<typename Robot>
void run_test(std::string graph_file_path, std::string scene_file_path, int resolution, std::string robot_name) {
    std::cout << "Running test for robot: " << robot_name << "\n";
    std::cout << "Creating environments from scene file: " << scene_file_path << "\n";
    std::vector<Environment<float>> h_envs = setup_environments(scene_file_path);
    std::cout << "Number of environments: " << h_envs.size() << "\n";
    std::cout << "Creating graph from file: " << graph_file_path << "\n";
    Graph g = read_graph_from_file(graph_file_path);
    std::cout << "Number of vertices: " << boost::num_vertices(g) << "\n";
    std::cout << "Number of edges: " << boost::num_edges(g) << "\n";
    std::vector<std::array<typename Robot::Configuration, 2>> edges_vec;

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
        for (int i = 0; i < Robot::dimension; i++) {
            
            edge[0][i] = g[src].states[i];
            edge[1][i] = g[tgt].states[i];
        }
        edges_vec.push_back(edge);
        
    }

    std::size_t num_edges = edges_vec.size();
    std::size_t num_envs = h_envs.size();
    std::cout << "Number of edges: " << num_edges << "\n";
    std::cout << "Number of environments: " << num_envs << "\n";
    
    std::vector<bool> results(num_edges * num_envs);
    batch_cc::batch_cc<Robot>(h_envs, edges_vec, resolution, results);

    for (int i = 0; i < num_edges; i++) {
        std::cout << "Edge " << i << ": ";
        for (int j = 0; j < num_envs; j++) {
            std::cout << results[i * num_envs + j] << " ";
        }
        std::cout << "\n";
    }

    // for each environment print number of spheres, cuboids, capsules
    for (int i = 0; i < num_envs; i++) {
        std::cout << "Environment " << i << ": ";
        std::cout << h_envs[i].num_spheres << " spheres, ";
        std::cout << h_envs[i].num_cuboids << " cuboids, ";
        std::cout << h_envs[i].num_capsules << " capsules\n\n";
    }

    // // Clean up the environments
    // for (auto env : h_envs) {
    //     delete env;
    // }
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
        run_test<robots::Panda>(graph_file_path, scene_file_path, resolution, robot_name);
    }
    else if (robot_name == "fetch") {
        run_test<robots::Fetch>(graph_file_path, scene_file_path, resolution, robot_name);
    }
    else {
        std::cout << "Unknown robot name: " << robot_name << "\n";
        return -1;
    }
    return 0;
}