import os, sys
import numpy as np
import tempfile
import xml.etree.ElementTree as ET

from envs.ant_env import AntEnv
from utils.maze_utils import construct_maze


MODEL_DIR = os.path.abspath('assets')

class AntMazeEnv(AntEnv):
    def __init__(self):
        sys.path.append(os.path.dirname(__file__))
        
        model_file = os.path.abspath(
            os.path.join(
                MODEL_DIR, 'ant.xml'
            )
        )

        file_path = self._load_maze(model_file)

        super(AntMazeEnv, self).__init__(file_path=file_path, ori_idx=6)


    def get_current_maze_obs(self,
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            maze_id=0,
            length=1,
            maze_height=0.5,
            maze_size_scaling=2,
            goal_rew=1.,
    ):  

        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span
        self._maze_id = maze_id
        self.length = length
        self.goal_rew = goal_rew

        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1 or structure[i][j] == 'g':
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + 1.0 * (2 * ray_idx + 1) / (2 * self._n_bins) * self._sensor_span
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self._sensor_range:
                        wall_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                    if first_seg["distance"] <= self._sensor_range:
                        goal_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                else:
                    assert False

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])

        return obs


    def step(self, action):
        inner_next_obs, inner_rew, done, info = self.env.step(action)
        next_obs = self.get_current_obs()
        x, y = self.wrapped_env.get_body_com("torso")[:2]
        # ref_x = x + self._init_torso_x
        # ref_y = y + self._init_torso_y
        info['outer_rew'] = 0
        info['inner_rew'] = inner_rew
        reward = self.coef_inner_rew * inner_rew
        minx, maxx, miny, maxy = self._goal_range
        if minx <= x <= maxx and miny <= y <= maxy:
            done = True
            reward += self.goal_rew
            info['rew_rew'] = 1  # we keep here the original one, so that the AvgReturn is directly the freq of success
        return Step(next_obs, reward, done, **info)


    def _load_maze(self, model_file, maze_height=0.5, maze_size_scaling=2):
        tree = ET.parse(model_file)
        worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self.MAZE_STRUCTURE = structure = construct_maze(maze_id=np.random.randint(0,10))

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1"
                    )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path) 
        return file_path

    def _find_robot(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling