# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# mass=5,
# rotational_mass=0.5,
# friction=[3, 10, 30, 50],
# thrust_x_pos=[-0.3, 0.3],
# color=[1, 1, 0],
# friction_function=lambda friction, vel: friction * (vel ** 3 / 10 + vel),
class boat:
    def __init__(
        self,
        time=0,
        pos=[0, 0, 0],
        vel_world=[0, 0, 0],
        throttle=[0, 0],
        color=[1, 1, 0],  # go beavs
        hullshape=None,
        friction=[3, 10, 30, 50],  # forwards, backwards, sideways, rotation
        thrust_x_pos=[-0.2, 0.2],
        mass=5,
        rotational_mass=0.5,
        thrust_function=lambda x: x / 20,
        friction_function=lambda friction, vel: friction * (vel ** 3 / 10 + vel),
        friction_function_rotation=lambda friction, vel: friction
        * (vel ** 3 / 10 + vel),
        max_num_history=100000,
    ):
        # kinematics
        self.time = time
        self.pos = {"x": pos[0], "y": pos[1], "az": pos[2]}
        self.vel_world = {"x": vel_world[0], "y": vel_world[1], "az": vel_world[2]}
        # physics
        self.friction = {
            "forwards": friction[0],
            "backwards": friction[1],
            "sideways": friction[2],
            "rotation": friction[3],
        }
        self.thrust_x_pos = thrust_x_pos
        self.mass = mass
        self.rotational_mass = rotational_mass
        self.thrust_function = thrust_function
        self.friction_function = friction_function
        self.friction_function_rotation = friction_function_rotation
        # control input
        self.throttle = throttle
        # appearance
        self.color = color
        if hullshape is None:
            hullshape = np.array([[-1, -1, 0, 1, 1], [-1, 1, 2, 1, -1]]) * [
                [0.35],
                [0.35],
            ]
        self.hullshape = np.array(hullshape)

        # history
        # x,y,az,vx,vy,vaz
        self.__history = np.zeros((max_num_history, 7))
        self.__history[0, :] = np.concatenate(
            ([self.time], self.pos_vec, self.vel_world_vec)
        )
        self.num_history_epochs = 1
        self.is_history_looped = False

        # x,y,az,vx,vy,vaz,ax,ay,aaz,throttleL,throttleR
        self.__history_of_updates = np.zeros((max_num_history, 9))
        self.__history_of_updates[0, :] = np.concatenate(
            ([self.time], self.pos_vec, self.vel_world_vec, self.throttle)
        )
        self.num_updates = 0
        self.is_history_of_updates_looped = False

    @property
    def history(self):
        """ Property to return the history """
        current_ind = self.num_history_epochs
        if self.is_history_looped:
            return np.vstack(
                (self.__history[current_ind:, :], self.__history[0:current_ind, :]),
            )
        else:
            return self.__history[0:current_ind, :]

    @property
    def history_of_updates(self):
        """ Property to return the history of updates """
        if (
            self.num_updates == 0 and not self.is_history_of_updates_looped
        ):  # boat hasn't moved yet
            return self.__history_of_updates[0, :]
        else:
            current_ind = self.num_updates
            if self.is_history_of_updates_looped:
                return np.vstack(
                    (
                        self.__history_of_updates[current_ind:, :],
                        self.__history_of_updates[0:current_ind, :],
                    ),
                )
            else:
                return self.__history_of_updates[0:current_ind, :]

    def _update_history(self):
        # t, x, y, az, vx, vy, vaz
        self.__history[self.num_history_epochs, :] = np.concatenate(
            ([self.time], self.pos_vec, self.vel_world_vec)
        )
        max_num_history = self.__history.shape[0]
        self.num_history_epochs += 1
        if self.num_history_epochs == max_num_history:
            self.num_history_epochs = 0
            self.is_history_looped = True

    def _update_history_of_updates(self):
        # t,x,y,az,vx,vy,vaz,throttleL,throttleR
        self.__history_of_updates[self.num_updates, :] = np.concatenate(
            ([self.time], self.pos_vec, self.vel_world_vec, self.throttle)
        )
        max_num_history = self.__history_of_updates.shape[0]
        self.num_updates += 1
        if self.num_updates == max_num_history:
            self.num_updates = 0
            self.is_history_of_updates_looped = True

    @property
    def pos_vec(self):
        """ Return a list with [x,y,az] in it """
        return [self.pos["x"], self.pos["y"], self.pos["az"]]

    @property
    def vel_world_vec(self):
        """ Return velocity of boat in world coordinates as list"""
        vel_dict = self.vel_world
        return [vel_dict["x"], vel_dict["y"], vel_dict["az"]]

    @staticmethod
    def local_to_world_coords(xy, local_az):
        """ converts from local coordinates to world coordinates """
        # * note these equations are not normal rotation matrix...
        # azimuth is off by 90 degrees and reversed to trig used trig identities
        # instead of subtracting pi/2 every time
        cos = np.cos(-local_az * np.pi / 180)
        sin = np.sin(-local_az * np.pi / 180)
        world_x = xy[0] * cos + xy[1] * -sin
        world_y = xy[0] * sin + xy[1] * cos
        return [world_x, world_y]

    @staticmethod
    def world_to_local_coords(xy, local_az):
        """ converts from world coordinates to local coordinates """
        # * note these equations are not normal rotation matrix...
        # azimuth is off by 90 degrees and reversed to trig used trig identities
        # instead of subtracting pi/2 every time
        cos = np.cos(local_az * np.pi / 180)
        sin = np.sin(local_az * np.pi / 180)
        local_x = xy[0] * cos + xy[1] * -sin
        local_y = xy[0] * sin + xy[1] * cos
        return [local_x, local_y]

    def update_position(self, time_step, num_dt, vel_water_function=lambda xy: (0, 0)):
        """ Update the position of the boat """
        self._update_history_of_updates()
        # for the num_dt to move
        dt = time_step / num_dt
        for t_step in range(num_dt):
            # -------------------- ACCELERATION FROM WATER CURRENTS ----------
            # convert water velocity from world to boat reference frame
            vel_water_world_x, vel_water_world_y = vel_water_function(
                [self.pos["x"], self.pos["y"]]
            )
            vel_relative_world_x = self.vel_world["x"] - vel_water_world_x
            vel_relative_world_y = self.vel_world["y"] - vel_water_world_y

            vel_relative_boat_x, vel_relative_boat_y = self.world_to_local_coords(
                [vel_relative_world_x, vel_relative_world_y], self.pos["az"]
            )

            # compute friction forces due to relative water velocities
            # positive force is pushing in positive direction
            friction_force_boat_x = self.friction_function(
                self.friction["sideways"], vel_relative_boat_x
            )

            if vel_relative_boat_y > 0:
                friction_force_boat_y = self.friction_function(
                    self.friction["forwards"], vel_relative_boat_y
                )
            else:
                friction_force_boat_y = self.friction_function(
                    self.friction["backwards"], vel_relative_boat_y
                )

            # compute acceleration in world reference frame due to friction
            accel_friction_boat_x = -friction_force_boat_x / self.mass
            accel_friction_boat_y = -friction_force_boat_y / self.mass

            # if acceleration due to friction over the dt flips the sign of the velocity
            # *this can happen over a large timestep, because we are assuming
            # constant friction over the timestep, even though it's slowing down
            # this could probably be more accurate with some iterative approach
            # but it would be slower

            is_change_sign_x = abs(accel_friction_boat_x * dt) > abs(
                vel_relative_boat_x
            )
            is_change_sign_y = abs(accel_friction_boat_y * dt) > abs(
                vel_relative_boat_y
            )
            # set acceleration so that final velocity is equal to 0
            # this precents overshoot
            # "hey, slow it down until it stops over this timestep"
            if is_change_sign_x:
                accel_friction_boat_x = -vel_relative_boat_x / dt
            if is_change_sign_y:
                accel_friction_boat_y = -vel_relative_boat_y / dt

            # convert acceleration to world coordinates
            accel_friction_world_x, accel_friction_world_y = self.local_to_world_coords(
                [accel_friction_boat_x, accel_friction_boat_y], self.pos["az"]
            )

            # -------------------- ACCELERATION FROM THRUSTERS ----------
            # compute thrust forces from each motor  ** torque == force_boat_az **
            # positive force is pushing in positive direction in pos and az
            thrust_force_boat_y = self.thrust_function(
                self.throttle[0]
            ) + self.thrust_function(self.throttle[1])

            accel_thrust_boat_y = thrust_force_boat_y / self.mass
            accel_thrust_boat_x = 0

            accel_thrust_world_x, accel_thrust_world_y = self.local_to_world_coords(
                [accel_thrust_boat_x, accel_thrust_boat_y], self.pos["az"]
            )

            # -------------------- COMBINE WORLD ACCELERATIONS ----------
            accel_total_world_x = accel_friction_world_x + accel_thrust_world_x
            accel_total_world_y = accel_friction_world_y + accel_thrust_world_y

            # store new time, positions and velocities
            self.vel_world["x"] += accel_total_world_x * dt
            self.vel_world["y"] += accel_total_world_y * dt

            self.pos["x"] += self.vel_world["x"] * dt
            self.pos["y"] += self.vel_world["y"] * dt

            # -------------------- ROTATE BOAT --------------
            # compute forces influencing boat azimuth
            friction_force_boat_az = self.friction_function_rotation(
                self.friction["rotation"], self.vel_world["az"] / 50
            )

            thrust_force_boat_az = (
                self.thrust_function(self.throttle[0]) / -self.thrust_x_pos[0]
                + self.thrust_function(self.throttle[1]) / -self.thrust_x_pos[1]
            )
            # compute and convert acceleration to velocity and position
            accel_azimuth = (
                thrust_force_boat_az - friction_force_boat_az
            ) / self.rotational_mass
            self.vel_world["az"] += accel_azimuth * dt
            self.pos["az"] += self.vel_world["az"] * dt
            if self.pos["az"] > 360:
                self.pos["az"] -= 360

            # -------------------- UPDATE HISTORY --------------
            # faster round
            self.time = int((self.time + dt) * (10 ** 6)) / (10.0 ** 6)
            # self.time = np.round(self.time + dt, 6)  # avoid 0.99999999 LSB errors
            self._update_history()

    def plot_history_line(self, ax, line_color="b", line_width=1):
        """ plot a line of where the boat has been """
        x_plot = self.history[:, 1]
        y_plot = self.history[:, 2]
        ax.plot(x_plot, y_plot, ".-", color=line_color, linewidth=line_width)

    def get_boat_polygon(self, boat_pos=None, scale=1):
        if boat_pos is None:
            boat_pos = self.pos
        # boat scalex, boat scale y
        if type(scale) != list:
            scale = [scale, scale]
        boat_shape = self.hullshape * np.array(scale).reshape(2, 1)

        # rotate to world coordinates
        azrad = np.deg2rad(-boat_pos["az"])
        R = np.array(
            [[np.cos(azrad), -1 * np.sin(azrad)], [np.sin(azrad), np.cos(azrad)]]
        )
        plot_points = R.dot(boat_shape) + np.array(
            [boat_pos["x"], boat_pos["y"]]
        ).reshape(2, -1)

        return plot_points

    def plot_boat_position(
        self,
        ax,
        scale=[0.25, 0.5],
        times_to_plot=None,
        zorder=10,
        alpha=0.9,
        face_colors=None,
    ):
        """ plot the boat positiona and orientation at times in vector """
        if times_to_plot is None:
            times_to_plot = [self.time]
        if face_colors is None:
            face_colors = [self.color] * len(times_to_plot)

        # get indices of times to plot
        all_times = self.history[:, 0]
        for t, boat_color in zip(times_to_plot, face_colors):
            if t <= self.time:
                # get index of data for time requested
                t_delta = np.round(all_times - t, 3)
                ind = np.argmin(np.abs(t_delta))  # round to ms
                # get boat data from history
                boat_pos = self.history[ind, [1, 2, 3]]
                boat_pos = {"x": boat_pos[0], "y": boat_pos[1], "az": boat_pos[2]}
                # scale boat size
                plotpoints = self.get_boat_polygon(boat_pos, scale)
                # add patch to axes
                ax.add_patch(
                    Polygon(
                        plotpoints.T,
                        fc=boat_color,
                        edgecolor="k",
                        zorder=zorder,
                        alpha=alpha,
                    )
                )


if __name__ == "__main__":

    def watervelfun(xy):
        if xy[0] > -1:
            return (-0.25, 0.25)
        else:
            return (-0.5, 0)

    myboat = boat(pos=[0, 0, 90])

    for t in range(14):
        if t == 1:
            myboat.throttle = [60, 80]
        elif t == 5:
            myboat.throttle = [80, 20]
        elif t == 8:
            myboat.throttle = [-100, 100]
        elif t == 10:
            myboat.throttle = [10, 10]

        myboat.update_position(1, 2, vel_water_function=watervelfun)

    fig, ax = plt.subplots()
    myboat.plot_history_line(ax, line_color="k", line_width=3)
    myboat.plot_boat_position(
        ax, scale=2, times_to_plot=[0, 5, 14], face_colors=["r", "g", (0, 0, 0)],
    )
    plt.axis([-10, 10, -10, 10])
    ax.set_aspect("equal", "box")
    plt.grid(True)

    fig2, ax2 = plt.subplots()
    t = myboat.history[:, 0]
    vx = myboat.history[:, 4]
    vy = myboat.history[:, 5]
    az = myboat.history[:, 3]
    plt.subplot(2, 1, 1)
    plt.plot(t, vx, color="r", label="x velocity")
    plt.plot(t, vy, color="b", label="y velocity")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t, az)
    plt.show()
