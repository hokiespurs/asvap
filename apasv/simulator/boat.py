# %%
import numpy as np


class boat:
    def __init__(
        self,
        time=0,
        pos=[0, 0, 0],
        vel_boat=[0, 0, 0],
        acc_boat=[0, 0, 0],
        throttle=[0, 0],
        color="y",
        hullshape=np.array([[-1, -1, 0, 1, 1], [-1, 1, 2, 1, -1]]),
        friction=[10, 2, 30, 50],  # sideways, forwards, backwards, rotation
        thrust_pos=[[-0.2, 0.2], [-0.4, -0.4]],
        mass=5,
        rotational_mass=0.5,
        thrust_function=lambda x: x / 20,
        max_num_history=10000,
    ):
        # kinematics
        self.time = time
        self.pos = {"x": pos[0], "y": pos[1], "az": pos[2]}
        self.vel_boat = {"x": vel_boat[0], "y": vel_boat[1], "az": vel_boat[2]}
        # physics
        self.friction = friction
        self.thrust_pos = thrust_pos
        self.mass = mass
        self.rotational_mass = rotational_mass
        self.thrust_function = thrust_function
        # control input
        self.throttle = throttle
        # appearance
        self.color = color
        self.hullshape = np.array(hullshape)

        # history
        # x,y,az,vx,vy,vaz
        self.__history = np.zeros((max_num_history, 7))
        self.__history[0, :] = np.concatenate(
            ([self.time], self.pos_world, self.vel_world)
        )
        self.num_history_epochs = 1

        # x,y,az,vx,vy,vaz,ax,ay,aaz,throttleL,throttleR
        self.__history_of_updates = np.zeros((max_num_history, 9))
        self.__history_of_updates[0, :] = np.concatenate(
            ([self.time], self.pos_world, self.vel_world, self.throttle)
        )
        self.num_updates = 1

    @property
    def history(self):
        """ Property to return the history """
        return self.__history[self.num_history_epochs, :]

    @property
    def history_of_updates(self):
        """ Property to return the history of updates """
        return self.__history_of_updates[self.num_updates, :]

    @property
    def vel_world(self):
        vel_world_x, vel_world_y = self.boat_to_world_coords(
            [self.vel_boat["x"], self.vel_boat["y"]], self.pos["az"]
        )
        return np.array([vel_world_x, vel_world_y, self.vel_boat["az"]])

    @property
    def pos_world(self):
        return np.array([self.pos["x"], self.pos["y"], self.pos["az"]])

    def update_position(self, dt, num_dt, vel_water_function=lambda xy: (0, 0)):
        """ Update the position of the boat """
        # for the num_dt to move
        # convert water velocity from world to boat reference frame

        # compute boat relative velocities

        # compute friction forces

        # compute thrust forces from each motor

        # update velocity of boat in (x,y,az) based on forces and dt

        # update boat position
        #  based on avg position/velocity from previous and current timestep
        pass

    @staticmethod
    def boat_to_world_coords(xy, boat_az):
        """ converts from boat coordinates to world coordinates """
        return [0, 0]  # ADD ACTUAL DCM MATRIX

    @staticmethod
    def world_to_boat_coords(xy, boat_az):
        """ converts from world coordinates to boat coordinates """
        return [0, 0]  # ADD ACTUAL DCM MATRIX

    def plot_history_line(self, ax):
        """ plot a line of where the boat has been """
        pass

    def plot_boat_position(self, ax, times_to_plot):
        """ plot the boat positiona and orientation at times in vector """
        pass


if __name__ == "__main__":
    myboat = boat()
