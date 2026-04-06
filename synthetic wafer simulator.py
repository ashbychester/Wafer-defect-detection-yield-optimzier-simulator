from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt


class DieState(IntEnum):
    OFF_WAFER = 0
    GOOD = 1
    DEFECTIVE = 2


class Wafer:

    def __init__(self, grid_size=64):

        self.grid_size = grid_size
        self.center = (grid_size - 1) / 2
        self.radius = grid_size // 2
        self.map = np.full((self.grid_size, self.grid_size), DieState.OFF_WAFER)

        while True:
            base = np.random.uniform(0.005, 0.01)
            middle = np.random.uniform(0.01, 0.02)
            edge = np.random.uniform(0.02, 0.04)

            if base < middle < edge:
                break

        self.base_defect_rate = base
        self.middle_defect_rate = middle
        self.edge_defect_rate = edge

        self.center_reg = np.random.uniform(0.5, 0.7) * self.radius
        self.edge_reg = np.random.uniform(0.8, 0.92) * self.radius

        self.y, self.x = np.ogrid[0:self.grid_size, 0:self.grid_size]
        self.dist = np.sqrt((self.center - self.x) ** 2 + (self.center - self.y) ** 2)

    def generate(self):
        self.map[:] = DieState.OFF_WAFER

        inner_waf = self.dist <= self.center_reg
        middle_waf = (self.dist > self.center_reg) & (self.dist <= self.edge_reg)
        edge_waf = (self.dist > self.edge_reg) & (self.dist <= self.radius)

        region = [(inner_waf, self.base_defect_rate),
                  (middle_waf, self.middle_defect_rate),
                  (edge_waf, self.edge_defect_rate)
                  ]

        for mask, defect_rate in region:
            self.map[mask] = DieState.GOOD
            rand_vals = np.random.rand(self.grid_size, self.grid_size)
            self.map[mask & (rand_vals < defect_rate)] = DieState.DEFECTIVE

    def yield_rate(self):
        good_dies = self.map == DieState.GOOD
        total_dies = self.map != DieState.OFF_WAFER
        return np.sum(good_dies) / np.sum(total_dies)

    def wafer_visualization(self):
        plt.imshow(self.map, cmap='viridis', origin='lower')
        plt.show()


class EdgeRing:

    def __init__(self, edge_dist, defect_rate=None):
        self.edge_dist = edge_dist
        if defect_rate is None:
            self.defect_rate = np.random.uniform(0.5, 0.7)
        else:
            self.defect_rate = defect_rate

    def apply(self, wafer):
        mask = (wafer.dist > self.edge_dist) & (wafer.dist < wafer.radius)
        rand_vals = np.random.rand(wafer.grid_size, wafer.grid_size)
        final_mask = mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class EdgeLoc:

    def __init__(self, edge_dist, defect_rate=None, circ_perc=None, angle=None):
        self.edge_dist = edge_dist
        self.defect_rate = np.random.uniform(0.5, 0.8) if defect_rate is None else defect_rate
        self.circ_perc = np.random.uniform(0.05, 0.15) if circ_perc is None else circ_perc
        self.angle = np.random.uniform(-np.pi, np.pi) if angle is None else angle

    def apply(self, wafer):
        theta = np.arctan2(wafer.y - wafer.center, wafer.x - wafer.center)
        angle2 = self.angle + (self.circ_perc * 2 * np.pi)
        edge_loc_mask = ((wafer.dist > self.edge_dist) & (wafer.dist < wafer.radius)
                         & (self.angle <= theta) & (theta <= angle2))
        rand_vals = np.random.rand(wafer.grid_size, wafer.grid_size)
        final_mask = edge_loc_mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Center:

    def __init__(self, center_dist, defect_rate=None):
        self.center_dist = center_dist
        self.defect_rate = np.random.uniform(0.5, 0.9) if defect_rate is None else defect_rate

    def apply(self, wafer):
        normalized_dist = wafer.dist / self.center_dist
        local_defect = self.defect_rate * (1 - normalized_dist)
        local_defect_prob = np.clip(local_defect, 0, 1)
        center_mask = (wafer.dist < self.center_dist)
        rand_vals = np.random.rand(wafer.grid_size, wafer.grid_size)
        final_mask = center_mask & (rand_vals < local_defect_prob) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Loc:

    def __init__(self):
        pass

    def apply(self, wafer):
        pass


class Scratch:
    def apply(self, wafer):
        pass


class Random:

    def __init__(self, defect_rate=None):
        self.defect_rate = np.random.uniform(0.2, 0.5) if defect_rate is None else defect_rate

    def apply(self, wafer):
        rand_vals = np.random.rand(wafer.grid_size, wafer.grid_size)
        final_mask = (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Donut:

    def __init__(self, radius=32, defect_rate=None, inner_rad=None, outer_rad=None):
        self.defect_rate = np.random.uniform(0.5, 0.9) if defect_rate is None else defect_rate
        self.inner_rad = np.random.uniform(0.3, 0.5)*radius if inner_rad is None else inner_rad
        self.outer_rad = np.random.uniform(0.7, 0.8)*radius if outer_rad is None else outer_rad

    def apply(self, wafer):
        donut_mask = (wafer.dist > self.inner_rad) & (wafer.dist < self.outer_rad)
        rand_vals = np.random.rand(wafer.grid_size, wafer.grid_size)
        final_mask = donut_mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE

class NearFull:

    def __init__(self, defect_rate=None):
        self.defect_rate = np.random.uniform(0.8, 0.99) if defect_rate is None else defect_rate

    def apply(self, wafer):
        rand_vals = np.random.rand(wafer.grid_size, wafer.grid_size)
        final_mask = (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


if __name__ == "__main__":
    a = Wafer()
    a.generate()
    print(a.yield_rate())
    a.wafer_visualization()

    e = Donut()
    e.apply(a)
    print(a.yield_rate())
    a.wafer_visualization()
