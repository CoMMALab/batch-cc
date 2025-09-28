"""
Collision check a set of edges (start_q, end_q) against a set of environments.
Environments: spheres, cuboids, capsules

Collision checking a single edge:
Discretize that edge based on resolution
For each q in the edge:
    do fk to find collision spheres for that q
    for each sphere -> loop through all environment obstacles and check if the sphere is in collision

"""