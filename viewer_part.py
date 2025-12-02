import numpy as np
import time

from panda3d.core import loadPrcFileData

loadPrcFileData('', 'notify-level error')
loadPrcFileData('', 'notify-output null')
from ursina import *

SCALE = 25  # Coordinate magnification

"""
Visually generate a map. Red represents the warehouse, blue represents the delivery point,
and the number above the task point is the randomly generated reward value. 
The robot's next target is marked in yellow, and the target it has passed is gray.
"""
class Viewer(object):
    def __init__(self):
        super().__init__()
        window.borderless = False
        window.fullscreen = False
        window.resizable = True
        window.size = (800, 600)  # Window settings

        self.viewer = Ursina()    # Registration window
        self.fresh = True

        # light
        DirectionalLight().look_at(Vec3(1, -1, -1))
        AmbientLight(color=color.rgba(100, 100, 100, 255))

        self.ground = self.create_plane()
        self.depot_viewers = np.array([], dtype=object)
        self.task_viewers = np.array([], dtype=object)
        self.point_viewers = np.array([], dtype=object)
        self.robot = self.create_robot()

        # camera
        self.cam_offset = Vec3(0, 40, -40)
        camera.position = self.ground.position + self.cam_offset
        camera.look_at(self.ground.position)

        # dynamic
        self.speed = 4.0                   # speed
        self.running = False               # move
        self.tik_time = time.time()        # Timestamp
        self.path = np.array([], dtype=int)
        self.target_index = np.array([], dtype=int)
        self.last_target = None
        self.target = None
        self.path_lines = np.array([], dtype=object)
        self.current_path_line = self.creat_path(0, 0, Vec3(0, 0, 0))

        self._setup_ui()

        # Unloading animation status
        self.unloading = False
        self.unload_start = 0.0            # Animation start time
        self.unload_dur = 0.8              # Unloading animation duration (seconds)
        self.unload_box = None             # "Cargo Box" entity
        self.unload_from = None            # Animation start point (world coordinate Vec3)
        self.unload_to = None              # Animation end point (world coordinate Vec3)

    def reset(self):
        self.fresh = True

        for entity_array in [self.depot_viewers, self.task_viewers, self.point_viewers, self.path_lines]:
            for e in entity_array:
                if e is not None and hasattr(e, "enabled"):
                    e.enabled = False
                    destroy(e)

        self.depot_viewers = np.array([], dtype=object)
        self.task_viewers = np.array([], dtype=object)
        self.point_viewers = np.array([], dtype=object)
        self.path_lines = np.array([], dtype=object)
        self.path = np.array([], dtype=int)
        self.target_index = np.array([], dtype=int)
        self.last_target = None
        self.target = None
        self.running = False
        self.tik_time = time.time()

        if self.robot is not None:
            destroy(self.robot)

        self.robot = self.create_robot()
        self.current_path_line = self.creat_path(0, 0, Vec3(0, 0, 0))
        self.viewer.step()

    def step(self):
        self.viewer.step()  # refresh display

    # Use two thin cubes to form a herringbone with an angle of ±35° around the X axis
    def _make_gable_roof(self, w, thickness, d, height, pos, clr, parent=None):
        roof_parent = Entity(position=pos, parent=parent)
        left = Entity(
            model='cube', color=clr, parent=roof_parent,
            scale=(w, thickness, d),
            position=(0, height/2, -d/4)
        )
        left.rotation_x = -35
        right = Entity(
            model='cube', color=clr, parent=roof_parent,
            scale=(w, thickness, d),
            position=(0, height/2,  d/4)
        )
        right.rotation_x = 35
        return roof_parent

    def _set_building_colors(self, ent, body_col, roof_col=None):  # Change wall and roof colors
        if roof_col is None:
            roof_col = body_col

        if hasattr(ent, "_parts"):
            if "body" in ent._parts and hasattr(ent._parts["body"], "color"):
                ent._parts["body"].color = body_col
            else:
                try:
                    ent.color = body_col
                except Exception:
                    pass

            roof = ent._parts.get("roof", None)
            if roof is not None:
                if hasattr(roof, "color"):
                    roof.color = roof_col
                for ch in getattr(roof, "children", []):
                    if hasattr(ch, "color"):
                        ch.color = roof_col
        else:
            try:
                ent.color = body_col
            except Exception:
                pass

    def _start_unload(self):  # Unloading animation
        if self.robot is None:
            return

        cube_size = 0.4            # Cube side length
        lateral_offset = 0.40       # The distance offset to the "right" side of the robot
        start_height = 0.35         # starting height
        end_height = 0.06           # ending height
        self.unload_dur = 0.6       # Animation duration

        # Calculate right direction
        right_vec = self.robot.right
        right_vec = Vec3(right_vec.x, 0, right_vec.z)
        if right_vec.length() < 1e-6:
            right_vec = Vec3(1, 0, 0)
        else:
            right_vec = right_vec.normalized()

        base = self.robot.world_position + right_vec * lateral_offset   # Right reference point
        start_world = Vec3(base.x, start_height, base.z)                # Drop start/end point
        end_world = Vec3(base.x, end_height, base.z)
        box = Entity(model='cube', color=color.rgb(230, 180, 120))      # goods
        box.world_scale = Vec3(cube_size, cube_size, cube_size)
        box.world_position = start_world

        self.unloading = True              # Unloading status
        self.unload_box = box
        self.unload_start = time.time()
        self.unload_from = start_world
        self.unload_to = end_world
        self.tik_time = time.time()        # Entering unloading: reset timer

    def _finish_arrival_and_advance(self):
        if self.target is not None:        # Mark visited
            self.set_visited(self.target)
        if len(self.target_index) > 0:
            self.target_index = np.delete(self.target_index, 0)
        if len(self.target_index) > 0:    # Continue to the next goal or end
            self.set_target(self.target_index[0])
        else:
            self.target = None
            self.running = False
            self.current_path_line.scale = (0.05, 0.02, 0)

    def _is_depot(self, ent):             # Determine whether it is a warehouse, for animation
        return hasattr(ent, "_parts") and ent._parts.get("kind") == "depot"

    def create_plane(self, tex_path='map.png', flip_v=True):               # map plane
        tex = load_texture(tex_path)
        if tex is None:
            return Entity(model='plane', color=color.light_gray,
                          scale=(SCALE, 1, SCALE),
                          position=(SCALE / 2, 0.0, SCALE / 2))
        ground = Entity(model='plane', texture=tex, color=color.white,
                        position=(SCALE / 2, 0.0, SCALE / 2),
                        scale=(SCALE, 1, SCALE))
        ground.texture_scale = (1, -1) if flip_v else (1, 1)
        return ground

    def create_depot(self, x, y):
        base = Entity(position=(x * SCALE, 0, y * SCALE))

        body = Entity(model='cube', parent=base,
                      color=color.rgb(230, 100, 100),
                      scale=(0.60, 0.27, 0.60),
                      position=(0, 0.175, 0))
        Entity(model='quad', parent=body, color=color.rgb(240, 240, 240),
               position=(0, -0.02, 0.41), scale=(0.40, 0.25))
        for i in range(5):
            Entity(model='quad', parent=body, color=color.rgb(210, 210, 210),
                   position=(0, -0.12 + i * 0.06, 0.415), scale=(0.40, 0.01))
        roof = self._make_gable_roof(
            w=0.58, thickness=0.03, d=0.58, height=0.12,
            pos=(0, 0.45, 0), clr=color.rgb(180, 40, 40), parent=base
        )
        try:
            Text(text="DEPOT", color=color.white, scale=0.36 * SCALE,
                 origin=(0, 0), position=(0, 0.60, 0), parent=base, billboard=True)
        except TypeError:
            Text(text="DEPOT", color=color.white, scale=0.36 * SCALE,
                 origin=(0, 0), position=(0, 0.60, 0), parent=base)
        base._parts = {"body": body, "roof": roof, "kind": "depot"}
        return base

    def create_task(self, x, y ,reward):
        base = Entity(position=(x * SCALE, 0, y * SCALE))
        base.scale = 0.75              # Uniformly scale down to 0.75 times the original size
        base.rotation_y = 60           # Rotate 20° around the y-axis
        body = Entity(model='cube', parent=base,
                      color=color.rgb(190, 220, 255),
                      scale=(0.45, 0.25, 0.45),
                      position=(0, 0.125, 0))
        Entity(model='quad', parent=body, color=color.rgb(160, 140, 120),
               position=(0, -0.02, 0.26), scale=(0.12, 0.18))
        Entity(model='quad', parent=body, color=color.rgb(240, 240, 255),
               position=(-0.12, 0.03, 0.26), scale=(0.10, 0.10))
        Entity(model='quad', parent=body, color=color.rgb(240, 240, 255),
               position=(0.12, 0.03, 0.26), scale=(0.10, 0.10))
        intensity = max(0.0, min(1.0, (reward - 5) / (20 - 5)))
        roof_color = color.rgb(int(80 * (1 - intensity)),
                               int(120 * (1 - intensity)),
                               255)
        roof = self._make_gable_roof(
            w=0.42, thickness=0.04, d=0.42, height=0.14,
            pos=(0, 0.28, 0), clr=roof_color, parent=base
        )
        try:
            Text(text=str(int(reward)), color=color.white, scale=0.5 * SCALE,
                 origin=(0, 0), position=(0, 1.0, 0), parent=base, billboard=True)
        except TypeError:
            Text(text=str(int(reward)), color=color.white, scale=0.5 * SCALE,
                 origin=(0, 0), position=(0, 1.0, 0), parent=base)

        base._parts = {"body": body, "roof": roof, "kind": "task", "reward": reward}
        return base

    def create_robot(self):
        body = Entity(model='cube', color=color.green,
                      scale=(0.4, 0.4, 0.6), position=(0, 0.2, 0))
        head = Entity(model='cube', parent=body, color=color.azure,
                      scale=(1.0, 0.5, 0.4), position=(0, 0.45, 0.18))
        cargo_size = 0.23      # Cargo size
        cargo_gap = -0.25
        back_dir = body.back
        back_dir = Vec3(back_dir.x, 0, back_dir.z)
        back_dir = back_dir.normalized() if back_dir.length() >= 1e-6 else Vec3(0, 0, -1)
        height_offset = 0.25
        offset_z = body.scale_z * 0.5 + cargo_size * 0.5 + cargo_gap
        cargo_world_pos = Vec3(body.world_x, cargo_size / 2 + height_offset, body.world_z) + back_dir * offset_z
        cargo = Entity(model='cube', color=color.rgb(210, 170, 120))
        cargo.world_parent = body
        cargo.world_scale = Vec3(cargo_size, cargo_size, cargo_size)
        cargo.world_position = cargo_world_pos

        body._parts = {"head": head}
        body._parts = getattr(body, "_parts", {})
        body._parts["cargo"] = cargo
        return body

    def creat_path(self, length, mid, direction):
        path = Entity(model='cube',
                      color=color.yellow,
                      scale=(0.05, 0.02, length),
                      position=mid,
                      rotation_y=np.degrees(np.arctan2(direction.x, direction.z)))
        return path

    def set_depots(self, depots):
        for x, y in depots:
            depot = self.create_depot(x, y)
            self.depot_viewers = np.append(self.depot_viewers, depot)
            self.point_viewers = np.append(self.point_viewers, depot)

    def set_tasks(self, tasks, rewards):
        for i, (x, y) in enumerate(tasks):
            reward = rewards[i]
            task = self.create_task(x, y, reward)
            self.task_viewers = np.append(self.task_viewers, task)
            self.point_viewers = np.append(self.point_viewers, task)

    # Visited points are grayed out
    def set_visited(self, point):
        self._set_building_colors(point, color.gray, color.gray)

    # The point you are about to go to turns yellow
    def set_target(self, target_index):
        self.target = self.point_viewers[target_index]
        self.rotate_robot(self.target)
        self._set_building_colors(self.target, color.yellow, color.yellow)

    def rotate_robot(self, target):
        if self.robot is None:
            return
        rx = self.target.position.x - self.robot.position.x
        ry = self.target.position.z - self.robot.position.z
        angle = np.degrees(np.arctan2(rx, ry))
        self.robot.rotation_y = angle  # Use the minus sign to ensure the correct direction

    def update_robot_pos(self, x, y):
        self.robot.position = (x, 0.15, y)

    def move_robot(self, path):
        if len(path) > len(self.path):
            new_path = path[len(self.path):]
        else:
            new_path = np.array([], dtype=int)

        # Update path record
        self.path = path
        self.target_index = new_path

        # If called for the first time
        if self.last_target is None and len(path) > 0:
            self.last_target = self.point_viewers[path[0]]
            self.update_robot_pos(self.last_target.x, self.last_target.z)
            self.target_index = np.delete(self.target_index, 0)

        # If there are new goals
        if len(self.target_index) > 0:
            self.set_target(self.target_index[0])
            self.running = True
            self.tik_time = time.time()

    def update(self):
        if self.robot is None:
            return

        # If in unloading animation: advance animation and continue forward when it ends
        if self.unloading and self.unload_box is not None:
            t = (time.time() - self.unload_start) / self.unload_dur
            if t >= 1.0:
                # Animation ends
                self.unload_box.enabled = False
                destroy(self.unload_box)
                self.unload_box = None
                self.unloading = False
                # Complete post-arrival processing & move on to the next destination
                self.tik_time = time.time()
                self._finish_arrival_and_advance()
            else:
                # Linear interpolation drop + slight scaling reduction
                p = lerp(self.unload_from, self.unload_to, t)
                s = 0.14 * (1.0 - 0.3 * t)  # From 0.14 to ~0.098
                self.unload_box.world_position = p
                self.unload_box.world_scale = Vec3(s, s, s)
            # Suspend movement during unloading
            self.viewer.step()
            return

        # General move logic
        if not self.running or self.target is None:
            return

        current = self.robot.position
        direction = self.target.position - current
        dist = direction.length()
        delta_time = time.time() - self.tik_time
        if delta_time > 0.05:
            delta_time = 0.05

        # Arriving at the target: first draw a fixed line segment & stop, then start the unloading animation
        if dist < delta_time * self.speed:
            self.robot.position = self.target.position

            if self.last_target is not None:
                start = self.last_target.position
                end = self.target.position
                mid = (start + end) / 2
                length = distance(start, end)
                line = self.creat_path(length, mid, end - start)
                self.path_lines = np.append(self.path_lines, line)

            self.last_target = self.target

            if self._is_depot(self.target):
                # Arrival at the depot
                self.tik_time = time.time()  # 重置时间，防止后续跳跃
                self._finish_arrival_and_advance()
                return
            else:
                # Arrival at mission point
                self._start_unload()
                return

        # Not arrived: Keep moving
        direction = direction.normalized()
        self.robot.position += direction * delta_time * self.speed
        self.tik_time = time.time()

        # current_path_line
        if self.last_target is not None:
            start = self.last_target.position
            end = self.robot.position
            mid = (start + end) / 2
            length = distance(start, end)
            self.current_path_line.scale = (0.05, 0.02, length)
            self.current_path_line.position = mid
            self.current_path_line.rotation_y = np.degrees(np.arctan2(direction.x, direction.z))

        self.viewer.step()

    def _setup_ui(self):    # Show title and legend
        self.title_text = Text(
            text="Task Planning for Autonomous Vehicle Delivery",
            parent=camera.ui,
            x=-.5, y=.47,
            scale=1.4, color=color.white
        )

        self.legend_panel = Entity(
            parent=camera.ui,
            position=(.46, .46)
        )
        Text("Legend", parent=self.legend_panel, x=-.02, y=.0, scale=0.9, color=color.white)

        def legend_row(idx, clr, label):
            y = -0.04 * (idx + 1)
            Entity(model='quad', color=clr, parent=self.legend_panel,
                   position=(-.02, y), scale=(.02, .02))
            Text(label, parent=self.legend_panel, x=.01, y=y + .005,
                 scale=0.8, color=color.rgba(230, 230, 230, 255))

        legend_row(0, color.red, "Depot")
        legend_row(1, color.blue, "Task")
        legend_row(2, color.gray, "Visited")
        legend_row(3, color.yellow, "Current Target / Path")
        legend_row(4, color.rgb(230, 180, 120), "Cargo")

    def hold(self):  # Finishing off the UI interaction
        Text("Press ESC or close window to exit",
             parent=camera.ui, x=-.25, y=-.47, scale=0.8,
             color=color.rgba(200, 200, 200, 255))

        def _input(key):
            if key in ('escape', 'q'):
                application.quit()

        self.viewer.input = _input

        self.viewer.run()