#!/usr/bin/env python3

import numpy as np

from constants import *


class EncoderReader:
    def __init__(self, bp, left_port, right_port):
        self.bp = bp
        self.left_port = left_port
        self.right_port = right_port

        self.reset()

    def reset(self):
        self.bp.offset_motor_encoder(
            self.left_port, self.bp.get_motor_encoder(self.left_port)
        )
        self.bp.offset_motor_encoder(
            self.right_port, self.bp.get_motor_encoder(self.right_port)
        )
        self.curr = self.read_encoders()

    def read_encoders(self):
        left = self.bp.get_motor_encoder(self.left_port)
        right = self.bp.get_motor_encoder(self.right_port)
        return np.array((left, right))

    def read_encoders_diff(self):
        next = self.read_encoders()
        diff = next - self.curr
        self.curr = next
        return diff


def transform(angle, dx, dy):
    return np.array(
        (
            (np.cos(angle), -np.sin(angle), dx),
            (np.sin(angle), np.cos(angle), dy),
            (0, 0, 1),
        )
    )


def state_to_pos(state):
    x = state[0][2]
    y = state[1][2]
    return np.array((x, y))


def state_to_angle_deg(state):
    rad = np.arctan2(state[1][0], state[0][0])
    deg = rad / np.pi * 180.0
    return deg


def print_state(state):
    DECIMAL_PLACES = 3
    angle = np.round(state_to_angle_deg(state), DECIMAL_PLACES)
    x = np.round(state[0][2], DECIMAL_PLACES)
    y = np.round(state[1][2], DECIMAL_PLACES)
    print("(x, y) = (%s, %s), theta = %s deg" % (x, y, angle))


def update_state(current_state, encoder_diffs):
    encoder_diffs_rad = encoder_diffs / 180 * np.pi
    arclengths = encoder_diffs_rad * WHEEL_RADIUS
    angle_diff = (arclengths[1] - arclengths[0]) / WHEEL_BASE_WIDTH

    distance = np.mean(arclengths)  # this is an approximation
    dx = np.cos(angle_diff) * distance
    dy = np.sin(angle_diff) * distance

    return np.matmul(current_state, transform(angle_diff, dx, dy))
