
import sys
import time
import numpy as np
import tkinter as tk
import pdb

WIDTH = 10
HEIGHT = 10
UNIT = 40

class Robot_Gridworld(tk.Tk, object):
    def __init__(self):
        super(Robot_Gridworld, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = 4
        self.n_features = 2  # numer of features in state
        self.title('Robot_GridWorld')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        self.build_gridworld()

    def build_gridworld(self):

        self.gridworld = tk.Canvas(self, bg='white',
                                width=WIDTH * UNIT,
                                height=HEIGHT * UNIT)


        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            self.gridworld.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            self.gridworld.create_line(x0, y0, x1, y1)


        origin = np.array([20, 20])


        robot_center = origin + np.array([0, UNIT * 9])
        self.robot = self.gridworld.create_oval(
            robot_center[0] - 15, robot_center[1] - 15,
            robot_center[0] + 15, robot_center[1] + 15,
            fill='yellow')


        bomb1_center = origin + np.array([UNIT * 4, UNIT*3])
        self.bomb1 = self.gridworld.create_rectangle(
            bomb1_center[0] - 15, bomb1_center[1] - 15,
            bomb1_center[0] + 15, bomb1_center[1] + 15,
            fill='red')

        bomb2_center = origin + np.array([UNIT * 2, UNIT*7])
        self.bomb2 = self.gridworld.create_rectangle(
            bomb2_center[0] - 15, bomb2_center[1] - 15,
            bomb2_center[0] + 15, bomb2_center[1] + 15,
            fill='red')

        bomb3_center = origin + np.array([UNIT * 7, UNIT*4])
        self.bomb3 = self.gridworld.create_rectangle(
            bomb3_center[0] - 15, bomb3_center[1] - 15,
            bomb3_center[0] + 15, bomb3_center[1] + 15,
            fill='red')

        bomb4_center = origin + np.array([UNIT * 7, UNIT*3])
        self.bomb4 = self.gridworld.create_rectangle(
            bomb4_center[0] - 15, bomb4_center[1] - 15,
            bomb4_center[0] + 15, bomb4_center[1] + 15,
            fill='red')


        treasure_center = origin + np.array([UNIT * 9, 0])
        self.treasure = self.gridworld.create_rectangle(
            treasure_center[0] - 15, treasure_center[1] - 15,
            treasure_center[0] + 15, treasure_center[1] + 15,
            fill='green')


        self.gridworld.pack()


    def reset(self):
        self.update()
        time.sleep(0.1)

        self.gridworld.delete(self.robot)

        origin = np.array([20, 20])
        robot_center = origin + np.array([0, UNIT * 9])

        self.robot = self.gridworld.create_oval(
            robot_center[0] - 15, robot_center[1] - 15,
            robot_center[0] + 15, robot_center[1] + 15,
            fill='yellow')

        return (np.array(self.gridworld.coords(self.robot)[:2]) - np.array(self.gridworld.coords(self.treasure)[:2])) / (
                    HEIGHT * UNIT)


    def step(self, action):
        s = self.gridworld.coords(self.robot)
        base_action = np.array([0, 0])
        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.gridworld.move(self.robot, base_action[0], base_action[1])

        next_state = self.gridworld.coords(self.robot)


        if next_state == self.gridworld.coords(self.treasure):
            reward = 100
            terminal = True
        elif next_state == self.gridworld.coords(self.bomb1):
            reward = -10
            terminal = True
        elif next_state == self.gridworld.coords(self.bomb2):
            reward = -10
            terminal = True
        elif next_state == self.gridworld.coords(self.bomb3):
            reward = -10
            terminal = True
        elif next_state == self.gridworld.coords(self.bomb4):
            reward = -10
            terminal = True
        else:
            reward = -0.1
            terminal = False

        next_s = (np.array(next_state[:2]) - np.array(self.gridworld.coords(self.treasure)[:2])) / (HEIGHT * UNIT)

        return next_s, reward, terminal


    def render(self):
        self.update()

