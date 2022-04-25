#!/usr/bin/env python3

from re import T
import sys
import time
import traceback
import numpy as np

import brickpi3

BP = brickpi3.BrickPi3()

LIGHT_SENSOR_PORT = BP.PORT_2
LOOP_DURATION = 0.01

LIGHT_THRESHOLD = 2500  # above this is black

if __name__ == "__main__":
    try:
        # Setup
        BP.reset_all()
        BP.set_sensor_type(LIGHT_SENSOR_PORT, BP.SENSOR_TYPE.NXT_LIGHT_ON)
        time.sleep(0.1)

        while True:
            start_time = time.perf_counter()
            light = BP.get_sensor(LIGHT_SENSOR_PORT)
            print(light)
            duration = time.perf_counter() - start_time
            time.sleep(max(0, LOOP_DURATION - duration))
            
    except KeyboardInterrupt:
        print("Interrupted")
        BP.reset_all()
    except Exception as e:
        traceback.print_exc()
        print("Encountered Exception:", e)
        BP.reset_all()