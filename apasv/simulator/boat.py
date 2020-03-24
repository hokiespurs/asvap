# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class boat:
    def __init__(
        self,
        time=0,
        pos=[0, 0, 0],
        vel_boat=[0, 0, 0],
        acc_boat=[0, 0, 0],
        throttle=[0, 0],
        color=[215 / 255, 63 / 255, 9 / 255],  # go beavs
        hullshape=np.array([[-1, -1, 0, 1, 1], [-1, 1, 2, 1, -1]]),
        friction=[2, 30, 10, 50],  # sideways, forwards, backwards, rotation
        thrust_x_pos=[-0.1, 0.1],
        mass=4,
        rotational_mass=0.5,
        thrust_function=lambda x: x / 20,
        friction_function=lambda friction, vel: -friction * vel * 3,
        max_num_history=10000,
    ):
        # kinematics
        self.time = time
        self.pos = {"x": pos[0], "y": pos[1], "az": pos[2]}
        self.vel_boat = {"x": vel_boat[0], "y": vel_boat[1], "az": vel_boat[2]}
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

        # control input
        self.throttle = throttle
        # appearance
        self.color = color
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
    def vel_world(self):
        """ Return velocity of boat in world coordinates as dictionary """
        vel_world_x, vel_world_y = self.local_to_world_coords(
            [self.vel_boat["x"], self.vel_boat["y"]], self.pos["az"]
        )
        return {"x": vel_world_x, "y": vel_world_y, "az": self.vel_boat["az"]}

    @property
    def vel_world_vec(self):
        """ Return velocity of boat in world coordinates as list"""
        vel_dict = self.vel_world
        return [vel_dict["x"], vel_dict["y"], vel_dict["az"]]

    @staticmethod
    def local_to_world_coords(xy, local_az):
        """ converts from local coordinates to world coordinates """

        azrad = np.deg2rad(-local_az)
        R_local_to_world = np.array(
            [[np.cos(azrad), -1 * np.sin(azrad)], [np.sin(azrad), np.cos(azrad)]]
        )
        world_xy = R_local_to_world.dot(np.array([xy[0], xy[1]]).reshape(2, 1))
        return [world_xy[0], world_xy[1]]

    @staticmethod
    def world_to_local_coords(xy, local_az):
        """ converts from world coordinates to local coordinates """
        azrad = -np.deg2rad(-local_az)
        R_world_to_local = np.array(
            [[np.cos(azrad), -1 * np.sin(azrad)], [np.sin(azrad), np.cos(azrad)]]
        )
        local_xy = R_world_to_local.dot(np.array([xy[0], xy[1]]).reshape(2, 1))
        return [local_xy[0], local_xy[1]]

    def update_position(self, time_step, num_dt, vel_water_function=lambda xy: (0, 0)):
        """ Update the position of the boat """
        # TODO Conservation of momentum is not right...
        # fast current goign into no current will have a show a crazy high acceleration
        self._update_history_of_updates()
        # for the num_dt to move
        dt = time_step / num_dt
        for t_step in range(num_dt):
            # convert water velocity from world to boat reference frame
            vel_water_world_x, vel_water_world_y = vel_water_function(
                [self.pos["x"], self.pos["y"]]
            )
            vel_water_boat_x, vel_water_boat_y = self.world_to_local_coords(
                [vel_water_world_x, vel_water_world_y], self.pos["az"]
            )

            # compute boat relative velocities
            vel_boat_relative_x = self.vel_boat["x"] - 0
            vel_boat_relative_y = self.vel_boat["y"] - 0

            # compute friction forces ** torque == force_boat_az **
            # positive force is pushing in positive direction
            friction_force_boat_x = self.friction_function(
                self.friction["sideways"], vel_boat_relative_x
            )
            if vel_boat_relative_y > 0:
                friction_force_boat_y = self.friction_function(
                    self.friction["forwards"], vel_boat_relative_y
                )
            else:
                friction_force_boat_y = self.friction_function(
                    self.friction["backwards"], vel_boat_relative_y
                )
            # * divided velocity by 50 because... reasons?
            # TODO Fix friction forces for azimuth, ensure friction not overshooting
            #   eg. 50m/s v_az w/ no throttle, corrects to -10 m/s v_az from friction
            friction_force_boat_az = self.friction_function(
                self.friction["rotation"], self.vel_boat["az"] / 200
            )
            # compute thrust forces from each motor  ** torque == force_boat_az **
            # positive force is pushing in positive direction in pos and az
            thrust_force_boat_y = self.thrust_function(
                self.throttle[0]
            ) + self.thrust_function(self.throttle[1])

            thrust_force_boat_az = (
                self.thrust_function(self.throttle[0]) / -self.thrust_x_pos[0]
                + self.thrust_function(self.throttle[1]) / -self.thrust_x_pos[1]
            )

            # Compute total forces acting on boat
            total_force_boat_x = friction_force_boat_x
            total_force_boat_y = thrust_force_boat_y + friction_force_boat_y
            total_force_boat_az = thrust_force_boat_az + friction_force_boat_az

            # calculate velocity and delta position of boat based on forces and dt

            new_vel_boat_x, delta_boat_pos_x = self.apply_boat_forces(
                self.vel_boat["x"], self.mass, total_force_boat_x, dt
            )
            new_vel_boat_y, delta_boat_pos_y = self.apply_boat_forces(
                self.vel_boat["y"], self.mass, total_force_boat_y, dt
            )
            new_vel_boat_az, delta_boat_pos_az = self.apply_boat_forces(
                self.vel_boat["az"], self.rotational_mass, total_force_boat_az, dt
            )
            # calculate relative boat position in world coordinates
            # note: uses the average azimuth over the timestep to go to world coords
            delta_world_pos_x, delta_world_pos_y = self.local_to_world_coords(
                [delta_boat_pos_x, delta_boat_pos_y], self.pos["az"],
            )

            # store new time, positions and velocities
            self.pos["x"] += delta_world_pos_x + vel_water_world_x * dt
            self.pos["y"] += delta_world_pos_y + vel_water_world_y * dt
            self.pos["az"] += delta_boat_pos_az  # daz_boat == daz_world
            if self.pos["az"] > 360:
                self.pos["az"] -= 360

            self.vel_boat["x"] = new_vel_boat_x
            self.vel_boat["y"] = new_vel_boat_y
            self.vel_boat["az"] = new_vel_boat_az

            self.time = np.round(self.time + dt, 6)  # avoid 0.99999999 LSB errors

            # update history
            self._update_history()

    @staticmethod
    def apply_boat_forces(vel_initial, mass, force, delta_time):
        accel = force / mass
        delta_velocity = accel * delta_time
        vel_final = vel_initial + delta_velocity
        # move boat based on average velocity over that time
        vel_average = vel_initial + delta_velocity / 2
        delta_position = vel_average * delta_time
        return (vel_final, delta_position)

    def plot_history_line(self, ax, line_color="b", line_width=1):
        """ plot a line of where the boat has been """
        x_plot = self.history[:, 1]
        y_plot = self.history[:, 2]
        ax.plot(x_plot, y_plot, color=line_color, linewidth=line_width)

    def plot_boat_position(
        self, ax, scale=[0.25, 0.5], times_to_plot=None, zorder=10, alpha=0.9
    ):
        """ plot the boat positiona and orientation at times in vector """
        if times_to_plot is None:
            times_to_plot = [self.time]

        # get indices of times to plot
        all_times = self.history[:, 0]
        for t in times_to_plot:
            if t <= self.time:
                # get index of data for time requested
                t_delta = np.round(all_times - t, 3)
                ind = np.argmin(np.abs(t_delta))  # round to ms
                # get boat data from history
                boat_pos = self.history[ind, [1, 2, 3]]
                # scale boat size
                if type(scale) != list:
                    scale = [scale, scale]
                boat_shape = self.hullshape * np.array(scale).reshape(2, 1)
                # rotate to world coordinates
                azrad = np.deg2rad(-boat_pos[2])
                R = np.array(
                    [
                        [np.cos(azrad), -1 * np.sin(azrad)],
                        [np.sin(azrad), np.cos(azrad)],
                    ]
                )
                plotpoints = R.dot(boat_shape) + np.array(
                    np.array([boat_pos[0], boat_pos[1]]).reshape(2, -1)
                )
                # add patch to axes
                ax.add_patch(
                    Polygon(
                        plotpoints.T,
                        fc=self.color,
                        edgecolor="k",
                        zorder=zorder,
                        alpha=alpha,
                    )
                )


if __name__ == "__main__":
    az = 30
    vx_world = 0.25
    vy_world = 0.5
    vx_boat, vy_boat = boat.world_to_local_coords([vx_world, vy_world], az)

    myboat = boat(pos=[0, 0, az])

    myboat.throttle = [100, 50]
    for _ in range(50):
        myboat.update_position(
            1, 100, vel_water_function=lambda xy: (vx_world, vy_world)
        )

    fig, ax = plt.subplots()
    myboat.plot_history_line(ax, line_color="k", line_width=3)
    myboat.plot_boat_position(ax, scale=[0.25, 0.35], times_to_plot=np.arange(0, 11, 1))
    plt.axis([-10, 30, -10, 10])
    ax.set_aspect("equal", "box")
    plt.grid(True)
    plt.show()

    t = myboat.history[:, 0]
    vx = myboat.history[:, 4]
    az = myboat.history[:, 3]
    plt.subplot(2, 1, 1)
    plt.plot(t, vx)
    plt.subplot(2, 1, 2)
    plt.plot(t, az)
    plt.show()

    boat_az = 45
    vx_boat, vy_boat = boat.world_to_local_coords([vx_world, vy_world], boat_az)
    print((vx_boat, vy_boat))
