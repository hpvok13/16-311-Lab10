#!/usr/bin/env python3

import sys
import time
import traceback
import numpy as np

import brickpi3

from localization import *
from movement import Movement
from odometry import *

BP = brickpi3.BrickPi3()

LEFT_MOTOR_PORT = BP.PORT_B
RIGHT_MOTOR_PORT = BP.PORT_C
LIGHT_SENSOR_PORT = BP.PORT_1
LOOP_DURATION = 0.01

LIGHT_THRESHOLD = 2500  # above this is black

# Use P control to follow the outer edge of the black line
LINE_P_MULTIPLIER = 0.05

STRAIGHT_MOTOR_DPS = np.array((180, 180))

LEFT_ARC_MOTOR_DPS = np.array((0, 180))

RIGHT_ARC_MOTOR_DPS = np.array((180, -180))

PROXIMITY_THRESHOLD = 0.5

BLACK_TIME_THRESHOLD = 0.1

WHITE_TIME_THRESHOLD = 0.03

WAYPOINT_1 = np.array((20, 0))

WAYPOINT_2 = np.array((20, 20))

def follow_line_until_black(er, state):
    time_in_black = 0
    loop_duration = 0
    while True:
        start_time = time.perf_counter()

        # Update state
        encoders_diff = er.read_encoders_diff()
        state = update_state(state, encoders_diff)

        # Read from sensors
        light = BP.get_sensor(LIGHT_SENSOR_PORT)

        # Follow line
        dps = STRAIGHT_MOTOR_DPS
        dps_add = None
        p_term = None

        light_diff = light - LIGHT_THRESHOLD
        if light_diff < 0:
            # white
            dps_add = np.array((1.0, -0.5))
            p_term = abs(light_diff) * LINE_P_MULTIPLIER * dps_add
            p_term = np.minimum(p_term, np.array((4, 0)))
            time_in_black = 0
        else:
            # black
            dps_add = np.array((-0.5, 1.0))
            p_term = abs(light_diff) * LINE_P_MULTIPLIER * dps_add
            time_in_black += loop_duration
            if time_in_black > BLACK_TIME_THRESHOLD:
                return
        dps = dps + p_term
        move.set_motor_dps(dps)

        duration = time.perf_counter() - start_time
        time.sleep(max(0, LOOP_DURATION - duration))

        loop_duration = time.perf_counter() - start_time

def follow_line_until_waypoint(er, state, waypoint):
    while True:
        start_time = time.perf_counter()

        # Update state
        encoders_diff = er.read_encoders_diff()
        state = update_state(state, encoders_diff)
        
        pos = state_to_pos(state)
        if np.linalg.norm(pos - waypoint) < PROXIMITY_THRESHOLD:
            return

        # Read from sensors
        light = BP.get_sensor(LIGHT_SENSOR_PORT)

        # Follow line
        dps = STRAIGHT_MOTOR_DPS
        dps_add = None
        p_term = None

        light_diff = light - LIGHT_THRESHOLD
        if light_diff < 0:
            # white
            dps_add = np.array((1.0, -1.0))
            p_term = abs(light_diff) * LINE_P_MULTIPLIER * dps_add
            p_term = np.minimum(p_term, np.array((4, 0)))
        else:
            # black
            dps_add = np.array((-1.0, 1.0))
            p_term = abs(light_diff) * LINE_P_MULTIPLIER * dps_add
        dps = dps + p_term
        move.set_motor_dps(dps)

        duration = time.perf_counter() - start_time
        time.sleep(max(0, LOOP_DURATION - duration))

def move_straight_until_white(er, move, state):
    # Move past black border until reach white
    loop_duration = 0
    time_in_white = 0
    move.set_motor_dps(STRAIGHT_MOTOR_DPS)
    while True:
        start_time = time.perf_counter()

        # Update state
        encoders_diff = er.read_encoders_diff()
        state = update_state(state, encoders_diff)

        light_diff = light - LIGHT_THRESHOLD

        if light_diff < 0:
            # White
            time_in_white += loop_duration
            if time_in_white > WHITE_TIME_THRESHOLD:
                return
        else:
            # Black
            time_in_white = 0

        duration = time.perf_counter() - start_time
        time.sleep(max(0, LOOP_DURATION - duration))

        loop_duration = time.perf_counter() - start_time

def move_straight_until_black(er, move, state):
    # Move past black border until reach white
    loop_duration = 0
    time_in_black = 0
    move.set_motor_dps(STRAIGHT_MOTOR_DPS)
    while True:
        start_time = time.perf_counter()

        # Update state
        encoders_diff = er.read_encoders_diff()
        state = update_state(state, encoders_diff)

        light_diff = light - LIGHT_THRESHOLD

        if light_diff < 0:
            # White
            time_in_black = 0
        else:
            # Black
            time_in_black += loop_duration
            if time_in_black > BLACK_TIME_THRESHOLD:
                return

        duration = time.perf_counter() - start_time
        time.sleep(max(0, LOOP_DURATION - duration))

        loop_duration = time.perf_counter() - start_time



if __name__ == "__main__":
    try:
        # Setup
        BP.reset_all()
        BP.set_sensor_type(LIGHT_SENSOR_PORT, BP.SENSOR_TYPE.NXT_LIGHT_ON)
        time.sleep(0.1)

        er = EncoderReader(BP, LEFT_MOTOR_PORT, RIGHT_MOTOR_PORT)

        move = Movement(BP, LEFT_MOTOR_PORT, RIGHT_MOTOR_PORT, er)

        # Main loop
        state = transform(0, 0, 0)
        while True:
            follow_line_until_black(er, state)

            move_straight_until_white(er, move, state)

            follow_line_until_waypoint(er, state, WAYPOINT_2)
            
            # Turn back toward goal
            move.turn(85, state)

            light = BP.get_sensor(LIGHT_SENSOR_PORT)
            light_diff = light - LIGHT_THRESHOLD
            black = light_diff >= 0

            assert not black, "Should be white at second waypoint"
            
            move_straight_until_black(er, move, state)
            
            follow_line_until_black(er, state)

            move_straight_until_white(er, move, state)

            follow_line_until_waypoint(er, state, np.array((0, 0)))
            
            # Orient to original pose
            move.turn(85, state)

            light = BP.get_sensor(LIGHT_SENSOR_PORT)
            light_diff = light - LIGHT_THRESHOLD
            black = light_diff >= 0

    except KeyboardInterrupt:
        print("Interrupted")
        BP.reset_all()
    except Exception as e:
        traceback.print_exc()
        print("Encountered Exception:", e)
        BP.reset_all()
