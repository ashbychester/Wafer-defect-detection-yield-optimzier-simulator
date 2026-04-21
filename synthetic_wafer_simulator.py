from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt


class DieState(IntEnum):
    OFF_WAFER = 0
    GOOD = 1
    DEFECTIVE = 2


class Wafer:

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.center = (grid_size - 1) / 2
        self.radius = grid_size // 2
        self.map = np.full((self.grid_size, self.grid_size), DieState.OFF_WAFER)
        self.y, self.x = np.ogrid[0:self.grid_size, 0:self.grid_size]
        self.dist = np.sqrt((self.center - self.x) ** 2 + (self.center - self.y) ** 2)

    def wafer_visualization(self):
        plt.imshow(self.map.astype(int), cmap='viridis', origin='lower')
        plt.show()


class Simulator:

    def __init__(self, grid_size=64, seed=None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

    def run(self, effects: list) -> Wafer:
        wafer = Wafer(self.grid_size)
        for eff in effects:
            eff.apply(wafer)
        return wafer


class WaferDataGenerator:

    def __init__(self, simulator: Simulator):
        self.simulator = simulator


class BaseRadialEffects:
    """
    First effect in the pipeline. Initializes all on-wafer dies to GOOD or DEFECTIVE
    based on radial region defect rates. Must run before any other effects.
    """

    def __init__(self, rng):
        self.rng = rng

        self.base_defect_rate = self.rng.uniform(0.005, 0.01)  # defect rate of center
        self.middle_defect_rate = self.rng.uniform(0.01, 0.02)  # defect rate of area between center and perimeter
        self.edge_defect_rate = self.rng.uniform(0.02, 0.04)  # defect rate of perimeter

        self.center_reg = self.rng.uniform(0.5, 0.7)  # ratio of wafer.radius for center region
        self.edge_reg = self.rng.uniform(0.8, 0.92)  # ratio of wafer.radius for perimeter/edge region

    def apply(self, wafer):
        wafer.map[:] = DieState.OFF_WAFER

        center_reg = self.center_reg * wafer.radius  # establishing where center region ends
        edge_reg = self.edge_reg * wafer.radius  # establishing where edge region begins

        # creating masks
        inner_waf = wafer.dist <= center_reg
        middle_waf = (wafer.dist > center_reg) & (wafer.dist <= edge_reg)
        edge_waf = (wafer.dist > edge_reg) & (wafer.dist <= wafer.radius)

        region = [(inner_waf, self.base_defect_rate),
                  (middle_waf, self.middle_defect_rate),
                  (edge_waf, self.edge_defect_rate)
                  ]

        for mask, defect_rate in region:
            wafer.map[mask] = DieState.GOOD
            rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
            wafer.map[mask & (rand_vals < defect_rate)] = DieState.DEFECTIVE


class YieldAnalyzer:
    """Has static method 'compute' for calculating Yield"""

    @staticmethod
    def compute(wafer):
        """Calculates Yield of wafer"""
        good_dies = wafer.map == DieState.GOOD
        total_dies = wafer.map != DieState.OFF_WAFER
        return np.sum(good_dies) / np.sum(total_dies)


class EdgeRing:

    def __init__(self, rng, defect_rate=None, edge_reg=None):
        self.rng = rng

        self.defect_rate = self.rng.uniform(0.5, 0.8) if defect_rate is None else defect_rate
        self.edge_reg = self.rng.uniform(0.8, 0.92) if edge_reg is None else edge_reg  # distance from center

    def apply(self, wafer):
        mask = (wafer.dist > self.edge_reg * wafer.radius) & (wafer.dist < wafer.radius)

        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class EdgeLoc:

    def __init__(self, rng, defect_rate=None, edge_reg=None, circ_perc=None, angle=None):
        self.rng = rng

        self.defect_rate = self.rng.uniform(0.5, 0.8) if defect_rate is None else defect_rate
        self.edge_reg = self.rng.uniform(0.8, 0.92) if edge_reg is None else edge_reg  # location of EdgeLoc defect
        self.circ_perc = self.rng.uniform(0.05, 0.15) if circ_perc is None else circ_perc  # how large EdgeLoc is
        self.angle = self.rng.uniform(-np.pi, np.pi) if angle is None else angle  # determines where EdgeLoc resides

    def apply(self, wafer):
        theta = np.arctan2(wafer.y - wafer.center, wafer.x - wafer.center)  # converts angle into degree coordinates
        angle2 = self.angle + (self.circ_perc * 2 * np.pi)  # 2nd angle where EdgeLoc defect ends
        edge_loc_mask = ((wafer.dist > self.edge_reg * wafer.radius) & (wafer.dist < wafer.radius)
                         & (self.angle <= theta) & (theta <= angle2))

        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = edge_loc_mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Center:

    def __init__(self, rng, center_dist=None, defect_rate=None, offset=0.05, offset_prob=0.5):
        self.rng = rng
        self.center_dist = self.rng.uniform(10, 20) if center_dist is None else center_dist  # how big center defect is
        self.defect_rate = self.rng.uniform(0.7, 0.9) if defect_rate is None else defect_rate
        self.offset = offset  # how offset the center is from regular
        self.offset_prob = offset_prob  # the chances of it happening, if larger then more likely

    def apply(self, wafer):
        dist = wafer.dist
        if self.rng.random > self.offset_prob:  # randomizing on whether offset will happen
            max_offset = self.offset * wafer.radius
            offset_x = self.rng.uniform(-max_offset, max_offset)
            offset_y = self.rng.uniform(-max_offset, max_offset)
            dist = np.sqrt((wafer.center + offset_x - wafer.x) ** 2 + (wafer.center + offset_y - wafer.y) ** 2)

        normalized_dist = dist / self.center_dist  # normalizing distance
        local_defect = self.defect_rate * (1 - normalized_dist)  # adjusting defect rate locally for a gradient
        local_defect_prob = np.clip(local_defect, 0, 1)  # ensuring values stay in range
        center_mask = (dist < self.center_dist)

        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = center_mask & (rand_vals < local_defect_prob) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Loc:

    def __init__(self, rng, defect_rate=None, location=None, x_rad=None, y_rad=None, angle=None):
        self.rng = rng
        self.defect_rate = self.rng.uniform(0.5, 0.9) if defect_rate is None else defect_rate
        while location is None:  # initializing location of loc defect
            x = self.rng.uniform(0, 64)
            y = self.rng.uniform(0, 64)
            dist = np.sqrt((x - 32) ** 2 + (y - 32) ** 2)
            if 5 < dist < 6:  # prevents ML from categorizing Loc as an EdgeLoc defect or a Center defect
                self.loc = (x, y)
                break
        else:
            self.location = location
        self.x_rad = self.rng.uniform(5, 12) if x_rad is None else x_rad  # radius in the x direction
        self.y_rad = self.rng.uniform(5, 12) if y_rad is None else y_rad  # radius in the y direction
        self.angle = self.rng.uniform(-np.pi, np.pi) if angle is None else angle  # angle of rotation of loc defect

    def apply(self, wafer):
        dx = self.location[0] - wafer.x
        dy = self.location[1] - wafer.y
        theta = self.angle

        x_rot = dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
        dist = np.sqrt((x_rot / self.x_rad) ** 2 + (y_rot / self.y_rad) ** 2)  # normalizing distance
        mask = dist <= 1

        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Scratch:

    def __init__(self, rng, defect_rate=None, angle=None, thickness=1, length=None, loc=None, wiggle=1):
        self.rng = rng
        self.defect_rate = self.rng.uniform(0.5, 0.9) if defect_rate is None else defect_rate
        self.angle = self.rng.uniform(-np.pi, np.pi) if angle is None else angle
        self.vector = (np.cos(self.angle), np.sin(self.angle))
        if loc is None:
            self.loc = (self.rng.uniform(0, 64), np.random.uniform(0, 64))
        else:
            self.loc = loc

        self.x = self.vector[0]
        self.y = self.vector[1]
        self.thickness = thickness
        self.length = np.random.uniform(32, 64) if length is None else length
        self.wiggle = wiggle

    def apply(self, wafer):
        line_points = []
        for i in range(int(self.length)):
            x_point = (self.loc[0] + i * self.x) + np.random.uniform(-self.wiggle, self.wiggle)
            y_point = (self.loc[1] + i * self.y) + np.random.uniform(-self.wiggle, self.wiggle)
            if not (0 <= x_point < wafer.grid_size):
                self.x *= -1
            if not (0 <= y_point < wafer.grid_size):
                self.y *= -1
            line_points.append((x_point, y_point))

        mask = np.zeros((wafer.grid_size, wafer.grid_size), dtype=bool)

        for x, y in line_points:
            new_x = int(round(x))
            new_y = int(round(y))
            for i in range(new_y - self.thickness, new_y + self.thickness + 1):
                for j in range(new_x - self.thickness, new_x + self.thickness + 1):
                    if 0 <= i < wafer.grid_size and 0 <= j < wafer.grid_size:
                        mask[i, j] = True
        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = mask & (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Random:

    def __init__(self, rng, defect_rate=None):
        self.rng = rng
        self.defect_rate = self.rng.uniform(0.2, 0.5) if defect_rate is None else defect_rate

    def apply(self, wafer):
        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class Donut:

    def __init__(self, rng, defect_rate=None, radius=32, inner_rad=None, outer_rad=None, falloff_perc=0.1,
                 center_prob=0.5, offset=0.05):
        self.rng = rng
        self.defect_rate = self.rng.uniform(0.5, 0.9) if defect_rate is None else defect_rate
        self.inner_rad = self.rng.uniform(0.2, 0.5) * radius if inner_rad is None else inner_rad  # inner radius
        self.outer_rad = self.rng.uniform(0.6, 0.8) * radius if outer_rad is None else outer_rad  # outer radius
        self.falloff_perc = falloff_perc  # percentage of falloff on the edges of donut for gradient
        self.center_prob = center_prob  # probability of an offset center
        self.offset = offset  # offset percentage of wafer radius

    def apply(self, wafer):
        dist = wafer.dist
        if np.random.random() > self.center_prob:  # randomizing prob of an offset center
            offset_prob = self.offset * wafer.radius
            offset_x = self.rng.uniform(-offset_prob, offset_prob)
            offset_y = self.rng.uniform(-offset_prob, offset_prob)
            dist = np.sqrt((wafer.center + offset_x - wafer.x) ** 2 + (wafer.center + offset_y - wafer.y) ** 2)

        falloff = wafer.radius * self.falloff_perc
        donut_mask = ((dist > self.inner_rad - falloff)
                      & (dist < self.outer_rad + falloff))  # mask enlarged by falloff for a gradient
        inner_defect = np.clip((dist - self.inner_rad - falloff) / falloff, 0, 1)
        outer_defect = np.clip((self.outer_rad + falloff - dist) / falloff, 0, 1)
        edge_fade = np.minimum(inner_defect, outer_defect)

        local_defect = self.defect_rate * edge_fade
        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = donut_mask & (rand_vals < local_defect) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


class NearFull:

    def __init__(self, rng, defect_rate=None):
        self.rng = rng
        self.defect_rate = np.random.uniform(0.8, 0.99) if defect_rate is None else defect_rate

    def apply(self, wafer):
        rand_vals = self.rng.random((wafer.grid_size, wafer.grid_size))
        final_mask = (rand_vals < self.defect_rate) & (wafer.map == DieState.GOOD)
        wafer.map[final_mask] = DieState.DEFECTIVE


if __name__ == "__main__":
    simulate = Simulator()
    effect = [BaseRadialEffects(simulate.rng)]
    a = simulate.run(effect)
    a.wafer_visualization()
