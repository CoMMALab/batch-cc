To build:
```
micromamba create -f environment.yaml
micromamba activate batch-cc
cmake -B build
cmake --build build
```
script usage
```
build/batch_cc <robot_name> <num_environments> <num_edges>
```
ex.
```
build/batch_cc panda 100 1000
```