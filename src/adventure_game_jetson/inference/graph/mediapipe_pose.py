from . import tools


num_node = 33
self_link = [(i, i) for i in range(num_node)]

edges_0_based = [
    (0, 11), (0, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

inward_ori_index = [(i + 1, j + 1) for (i, j) in edges_0_based]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode="spatial"):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            return tools.get_spatial_graph(num_node, self_link, inward, outward)
        raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")

