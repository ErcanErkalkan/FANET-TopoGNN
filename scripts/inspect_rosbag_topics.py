from __future__ import annotations

import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader


def main() -> None:
    parser = argparse.ArgumentParser(description="List ROS bag topics without deserializing messages.")
    parser.add_argument("bags", nargs="+", type=Path)
    args = parser.parse_args()
    for bag in args.bags:
        print(f"\n[{bag.name}]")
        with AnyReader([bag]) as reader:
            for connection in sorted(reader.connections, key=lambda item: item.topic):
                count = getattr(connection, "msgcount", "?")
                print(f"{connection.topic}\t{connection.msgtype}\t{count}")


if __name__ == "__main__":
    main()
