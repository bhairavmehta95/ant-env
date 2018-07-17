import os, sys
import numpy as np
import tempfile
import xml.etree.ElementTree as ET

from maze_utils import construct_maze

from ant_env import AntEnv


class AntMazeEnv(AntEnv):
    def __init__(self):
        sys.path.append(os.path.dirname(__file__))
        
        model_file = '/home/bhairav/coding/ant-env/models/ant.xml'
        file_path = self._load_maze(model_file)

        super(AntMazeEnv, self).__init__(file_path=file_path)


    def _load_maze(self, model_file, maze_height=0.5, maze_size_scaling=2):
        tree = ET.parse(model_file)
        worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self.MAZE_STRUCTURE = structure = construct_maze(maze_id=5)

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