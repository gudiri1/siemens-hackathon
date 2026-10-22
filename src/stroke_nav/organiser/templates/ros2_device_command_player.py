#!/usr/bin/env python3
"""Stream a planned trajectory to the organiser's EndoNav simulator as DeviceCommand setpoints.

Runs INSIDE the organiser's ROS 2 Jazzy environment (needs rclpy + catforce_interfaces); it was
written against HCMR-EndoNavSimDocker/src/catforce_ros_bridge/ROS2_INTERFACE.md and has not been run
outside that stack. Use with the dual-device page (catheter_guidewire_ros2.html), which latches each
setpoint and tracks it at the UI feed/spin rate. Disable OpenNG3's joint_state_follower first so it
does not publish competing commands.

    python3 ros2_device_command_player.py device_commands.csv [--rate-scale 1.0]

CSV columns: t_s, device (guidewire|catheter), insertion_setpoint_mm, rotation_setpoint_deg,
insertion_velocity_mm_s, rotation_velocity_deg_s, frames, enabled
"""

import argparse
import csv
import time

import rclpy
from catforce_interfaces.msg import DeviceCommand
from rclpy.node import Node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--rate-scale", type=float, default=1.0, help=">1 plays slower")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    rclpy.init()
    node = Node("stroke_nav_device_command_player")
    pubs = {d: node.create_publisher(DeviceCommand, f"/endovascular/{d}/command", 10) for d in ("guidewire", "catheter")}
    start = time.monotonic()
    try:
        for row in rows:
            due = start + float(row["t_s"]) * args.rate_scale
            while time.monotonic() < due:
                rclpy.spin_once(node, timeout_sec=0.002)
            msg = DeviceCommand()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.insertion_setpoint = float(row["insertion_setpoint_mm"])
            msg.rotation_setpoint = float(row["rotation_setpoint_deg"])
            msg.insertion_velocity = float(row["insertion_velocity_mm_s"])
            msg.rotation_velocity = float(row["rotation_velocity_deg_s"])
            msg.frames = int(row["frames"])
            msg.enabled = True
            pubs[row["device"]].publish(msg)
        node.get_logger().info("trajectory finished; devices keep tracking the last setpoint")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
