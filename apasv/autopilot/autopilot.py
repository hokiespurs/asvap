import numpy as np


class autopilot:
    """ Base Autopilot Class  """

    def __init__(self, survey_lines):
        self.survey_lines = survey_lines
        self.current_line = 0
        self.line_finished = True  # flag to make sure the boat goes in order
        self.mission_complete = False

    def update_line_status(self, pos_xy):
        line = self.survey_lines[self.current_line]

        # check if boat is past the last point, go to next point
        if self.line_finished is False:
            if self.current_line == len(self.survey_lines):
                self.mission_complete = True
                return

            downline_pos_backwards, _ = self.xy_to_downline_offline(
                pos_xy, [line["P2x"], line["P2y"]], [line["P1x"], line["P1y"]]
            )
            if downline_pos_backwards < 0:
                self.line_finished = True
                self.current_line += 1

                if self.current_line == len(self.survey_lines):
                    self.mission_complete = True
                    return

        line = self.survey_lines[self.current_line]
        # if a line was just finished, check if it's on the new line yet
        if self.line_finished is True:
            downline_pos, _ = self.xy_to_downline_offline(
                pos_xy, [line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]]
            )
            if downline_pos > 0:
                self.line_finished = False

    @staticmethod
    def xy_to_downline_offline(xy, p1, p2, do_vel=False):
        # TODO this code is duplicated from mission... move to a "common.py" folder
        az = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        if do_vel:
            downline = (xy[:, 0]) * np.cos(az) + (xy[:, 1]) * np.sin(az)
            offline = (xy[:, 0]) * -np.sin(az) + (xy[:, 1]) * np.cos(az)
        else:
            downline = (xy[:, 0] - p1[0]) * np.cos(az) + (xy[:, 1] - p1[1]) * np.sin(az)
            offline = (xy[:, 0] - p1[0]) * -np.sin(az) + (xy[:, 1] - p1[1]) * np.cos(az)

        return (downline, offline)
