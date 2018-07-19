import os, sys
import numpy as np
import tempfile
import xml.etree.ElementTree as ET
import math
from ctypes import byref

from mujoco_py import MjViewer, MjModel, mjcore, mjlib, \
    mjextra, glfw
from mujoco_py.mjlib import mjlib

from envs.ant_env import AntEnv
from utils.maze_utils import construct_maze
from utils.embedded_viewer import EmbeddedViewer

APPLE = 0
BOMB = 1

MODEL_DIR = os.path.abspath('assets')

class GatherViewer(MjViewer):
    def __init__(self, env):
        self.env = env
        super(GatherViewer, self).__init__()
        green_ball_model = MjModel(os.path.abspath(
            os.path.join(
                MODEL_DIR, 'green_ball.xml'
            )
        ))
        self.green_ball_renderer = EmbeddedViewer()
        self.green_ball_model = green_ball_model
        self.green_ball_renderer.set_model(green_ball_model)
        red_ball_model = MjModel(os.path.abspath(
            os.path.join(
                MODEL_DIR, 'red_ball.xml'
            )
        ))
        self.red_ball_renderer = EmbeddedViewer()
        self.red_ball_model = red_ball_model
        self.red_ball_renderer.set_model(red_ball_model)

    def start(self):
        super(GatherViewer, self).start()
        self.green_ball_renderer.start(self.window)
        self.red_ball_renderer.start(self.window)

    def stop_viewer(self):
        if self.env.viewer:
            self.env.viewer.finish()

    def render(self):
        super(GatherViewer, self).render()
        tmpobjects = mjcore.MJVOBJECTS()
        mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        for obj in self.env.objects:
            x, y, typ = obj
            # print x, y
            qpos = np.zeros_like(self.green_ball_model.data.qpos)
            qpos[0, 0] = x
            qpos[1, 0] = y
            if typ == APPLE:
                self.green_ball_model.data.qpos = qpos
                self.green_ball_model.forward()
                self.green_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.green_ball_renderer.objects)
            else:
                self.red_ball_model.data.qpos = qpos
                self.red_ball_model.forward()
                self.red_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.red_ball_renderer.objects)
        mjextra.append_objects(tmpobjects, self.objects)
        mjlib.mjv_makeLights(
            self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))

        try:
            import OpenGL.GL as GL
        except:
            return

        def draw_rect(x, y, width, height):
            # start drawing a rectangle
            GL.glBegin(GL.GL_QUADS)
            # bottom left point
            GL.glVertex2f(x, y)
            # bottom right point
            GL.glVertex2f(x + width, y)
            # top right point
            GL.glVertex2f(x + width, y + height)
            # top left point
            GL.glVertex2f(x, y + height)
            GL.glEnd()

        def refresh2d(width, height):
            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

        GL.glLoadIdentity()
        width, height = glfw.get_framebuffer_size(self.window)
        refresh2d(width, height)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_BLEND)

        GL.glColor4f(0.0, 0.0, 0.0, 0.8)
        draw_rect(10, 10, 300, 100)

        apple_readings, bomb_readings = self.env.get_readings()
        for idx, reading in enumerate(apple_readings):
            if reading > 0:
                GL.glColor4f(0.0, 1.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 10, 5, 50)
        for idx, reading in enumerate(bomb_readings):
            if reading > 0:
                GL.glColor4f(1.0, 0.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 60, 5, 50)


class AntGatherEnv(AntEnv):
    def __init__(self,
            n_apples=8,
            n_bombs=8,
            activity_range=10.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=2*math.pi,
            coef_inner_rew=0.,
            dying_cost=-10,
            max_steps=500
    ):

        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.dying_cost = dying_cost
        self.objects = []
        self.max_steps = max_steps

        model_file = os.path.abspath(
            os.path.join(
                MODEL_DIR, 'ant.xml'
            )
        )

        file_path = self._load_env(model_file)

        super(AntGatherEnv, self).__init__(file_path=file_path, ori_idx=6)


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.step_count = 0

        com = self.get_body_com("torso")
        x, y = com[:2]
        reward = 0
        new_objs = []

        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                else:
                    reward = reward - 1
            else:
                new_objs.append(obj)

        self.objects = new_objs
        state = self.state_vector()
        done = self._calculate_terminal(state)
        
        return self.get_current_obs(), reward, done, None


    def reset(self):
        self.step_count = 0

        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

        super(AntEnv, self).reset()
        
        return self.get_current_obs()


    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            self.get_viewer().render()
            data, width, height = self.get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        elif mode == 'human':
            self.get_viewer().loop_once()
        if close:
            self.get_viewer().stop_viewer()


    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()  # overwrite this for Ant!

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb; ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings


    def get_current_obs(self):
        obs = np.concatenate([
            self._get_obs(),
            np.array([self.get_readings()]).flatten()
        ])

        return obs


    def get_viewer(self):
        if self.viewer is None:
            self.viewer = GatherViewer(self)
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer


    def _load_env(self, model_file):
        tree = ET.parse(model_file)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)

        return file_path


    def _calculate_terminal(self, state):
        return len(self.objects) == 0 or \
            self.step_count > self.max_steps or \
            not np.isfinite(state).any() \
            or state[2] >= 0.2 or state[2] <= 1.0