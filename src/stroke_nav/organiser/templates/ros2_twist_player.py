#!/usr/bin/env python3
"""Replay a planned trajectory through the full OpenNG3 stack as autonomous Twist commands.

Subclasses the organiser's ``AlgorithmControllerBase`` (HCMR-EV-OpenNG3/src/bridge/scripts/
algorithm_control_template.py), which publishes on /control/autonomous/module_<n>/cmd_vel, lets
the arbiter hand the module over, and republishes from a watchdog. Runs INSIDE their ROS 2 Jazzy
environment; not executed outside that stack.

    python3 ros2_twist_player.py twists.csv --template-dir /path/to/HCMR-EV-OpenNG3/src/bridge/scripts

CSV columns: t_s, module (1 = guidewire, 2 = catheter), topic, linear_x_mm_s, angular_z_rad_s.
The bridge clamps to 20 mm/s and 5 rad/s; the RL action space allows 10 mm/s and 1 rad/s.
"""

import argparse
import bisect
import csv
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--template-dir", required=True, help="folder containing algorithm_control_template.py")
    args = ap.parse_args()
    sys.path.insert(0, args.template_dir)
    import rclpy
    from algorithm_control_template import AlgorithmControllerBase
    from geometry_msgs.msg import Twist

    table = {1: [], 2: []}
    for row in csv.DictReader(open(args.csv, encoding="utf-8")):
        table[int(row["module"])].append((float(row["t_s"]), float(row["linear_x_mm_s"]), float(row["angular_z_rad_s"])))
    times = {m: [r[0] for r in rows] for m, rows in table.items()}
    end_t = max(r[-1][0] for r in table.values() if r)

    class TrajectoryPlayer(AlgorithmControllerBase):
        def __init__(self):
            super().__init__("stroke_nav_twist_player")
            self.t0 = time.monotonic()

        def compute_command(self, module_id, feedback):
            t = time.monotonic() - self.t0
            if t > end_t or not table.get(module_id):
                return None  # explicit zero Twist from the base class
            i = max(bisect.bisect_right(times[module_id], t) - 1, 0)
            _, v, w = table[module_id][i]
            msg = Twist()
            msg.linear.x = v  # mm/s (OpenNG3 native unit)
            msg.angular.z = w  # rad/s
            return msg

    rclpy.init()
    node = TrajectoryPlayer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
