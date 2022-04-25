#!/usr/bin/env python3

import time
import numpy as np

from odometry import *


class Movement:
    DEBUG = True
    LOOP_SLEEP_DURATION = 0.01

    def __init__(self, bp, left_port, right_port, er):
        self.bp = bp
        self.left_port = left_port
        self.right_port = right_port
        self.er = er

    def set_motor_powers(self, powers):
        self.bp.set_motor_power(self.left_port, powers[0])
        self.bp.set_motor_power(self.right_port, powers[1])

    def set_motor_dps(self, dps):
        self.bp.set_motor_dps(self.left_port, dps[0])
        self.bp.set_motor_dps(self.right_port, dps[1])

    def turn(self, angle, state, raw=False):
        ANGLE_P_MULTIPLIER = 0.01
        ANGLE_D_MULTIPLIER = 0.15
        POWER_P_MULTIPLIER = 0.02
        KEEP_D_DIFFERENCES = 2
        TARGET_POWER = 20.0

        if not raw:
            angle %= 360
            if angle > 180:
                angle -= 360
            if angle < -180:
                angle += 360

        if abs(angle) < 1e-6:
            return

        self.er.reset()

        angle_rad = angle / 180.0 * np.pi
        arclength = angle_rad * (WHEEL_BASE_WIDTH / 2.0)
        wheel_angle_rad = arclength / WHEEL_RADIUS
        wheel_angle_deg = wheel_angle_rad * 180.0 / np.pi
        angle_offsets = np.array((-wheel_angle_deg, wheel_angle_deg))

        signs = np.sign(angle_offsets)

        curr_powers = np.array((0, 0))

        prev_angle_d_differences = []
        for _ in range(KEEP_D_DIFFERENCES):
            prev_angle_d_differences.append(np.array((0, 0)))

        while True:
            start_time = time.perf_counter()

            angles = self.er.read_encoders()

            # Update state
            encoders_diff = self.er.read_encoders_diff()
            state = update_state(state, encoders_diff)

            ts = angles / angle_offsets
            t = np.mean(ts)

            # If at destination, stop
            if t >= 1:
                self.set_motor_powers([0, 0])
                return

            # Apply feedback control to make wheels turn at a constant speed ratio
            desired_angles = t * angle_offsets
            curr_angle_differences = desired_angles - angles

            # P
            angle_difference_p_term = ANGLE_P_MULTIPLIER * curr_angle_differences

            # D
            prev_angle_d_differences.append(curr_angle_differences)
            prev_angle_d_differences = prev_angle_d_differences[1:]
            d_angle_differences = np.diff(prev_angle_d_differences, axis=0)
            avg_d_angle_differences = np.mean(d_angle_differences, axis=0)
            angle_difference_d_term = ANGLE_D_MULTIPLIER * avg_d_angle_differences

            angle_term = angle_difference_p_term + angle_difference_d_term

            # Apply feedback control to reach a desired average power
            abs_powers = np.absolute(curr_powers)
            avg_abs_power = np.mean(abs_powers)
            power_diff = TARGET_POWER - avg_abs_power
            power_difference_term = POWER_P_MULTIPLIER * power_diff * signs

            curr_powers = curr_powers + angle_term + power_difference_term

            if self.DEBUG:
                print("angles:", angles)
                print("desired_angles:", desired_angles)
                print("curr_angle_differences:", curr_angle_differences)
                print("angle_difference_p_term:", angle_difference_p_term)
                print("angle_difference_d_term:", angle_difference_d_term)
                print("power_difference_term:", power_difference_term)
                print("curr_powers:", curr_powers)
                print("------")

            self.set_motor_powers(curr_powers)


            duration = time.perf_counter() - start_time
            time.sleep(max(0, self.LOOP_SLEEP_DURATION - duration))

    def move_straight(self, distance):
        TARGET_POWER = 30
        ANGLE_P_MULTIPLIER = 0.01
        ANGLE_D_MULTIPLIER = 0.5
        POWER_P_MULTIPLIER = 0.03

        self.er.reset()
        state = transform(0, 0, 0)

        goal = np.array((distance, 0))

        curr_powers = np.array((0, 0))

        prev_angle_diff = 0
        while True:
            start_time = time.perf_counter()

            # Update state
            encoders_diff = self.er.read_encoders_diff()
            state = update_state(state, encoders_diff)

            # Compute desired change in angle
            pos = state_to_pos(state)
            print(pos)
            angle = state_to_angle_deg(state)

            goal_diff = goal - pos
            desired_angle = np.arctan2(
                goal_diff[1], goal_diff[0]
            )  # pointing towards goal

            angle_diff = desired_angle - angle
            d_angle_diff = angle_diff - prev_angle_diff
            prev_angle_diff = angle_diff

            # If at or past destination, stop
            if goal_diff[0] <= 0:
                self.set_motor_powers((0, 0))
                return

            # Angle PID
            angle_p_term = ANGLE_P_MULTIPLIER * angle_diff * np.array((-1, 1))
            angle_d_term = ANGLE_D_MULTIPLIER * d_angle_diff * np.array((-1, 1))
            angle_term = angle_p_term + angle_d_term

            # Power PID
            target_power = TARGET_POWER
            abs_powers = np.absolute(curr_powers)
            avg_abs_power = np.mean(abs_powers)
            power_diff = target_power - avg_abs_power
            power_p_term = POWER_P_MULTIPLIER * power_diff * np.array((1, 1))
            power_term = power_p_term

            # Update powers
            curr_powers = curr_powers + angle_term + power_term
            self.set_motor_powers(curr_powers)

            if self.DEBUG:
                print("pos:", pos)
                print("angle:", angle)
                print("goal:", goal)
                print("desired_angle:", desired_angle)
                print("goal_diff:", goal_diff)
                print("angle_diff:", angle_diff)

                print("angle_p_term:", angle_p_term)
                print("angle_d_term:", angle_d_term)
                print("power_p_term:", power_p_term)

                print("curr_powers:", curr_powers)
                print("---")

            duration = time.perf_counter() - start_time
            time.sleep(max(0, self.LOOP_SLEEP_DURATION - duration))
