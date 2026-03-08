#!/usr/bin/env python3
"""
Standalone desk height display and sit/stand test script.

Connects to a BLE standing desk via DeskClient, displays real-time
height updates, and accepts interactive sit/stand commands.

Usage:
    python scripts/test_desk.py [--config CONFIG_PATH] [--sitstand-path PATH]

Commands:
    s - Sit (move desk down)
    t - Stand (move desk up)
    h - Print current height
    q - Quit
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desk.desk_client import DeskClient, DeskState, DeskPosition


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    import yaml

    path = Path(__file__).parent.parent / config_path
    if not path.exists():
        print(f"Warning: Config file not found at {path}, using defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


async def read_command() -> str:
    """Read a single-line command without blocking the async loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, "\n> ")


async def main():
    parser = argparse.ArgumentParser(description="Test desk height display and sit/stand commands")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--sitstand-path",
        type=str,
        default=None,
        help="Path to sitstand repo (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    desk_cfg = config.get("desk", {})
    sitstand_path = args.sitstand_path or desk_cfg.get("sitstand_path", "../sitstand")

    print("Desk Control Test")
    print("=" * 40)

    # Create and connect
    desk = DeskClient(sitstand_path=sitstand_path, enabled=True)

    # Register state change callback
    async def on_state_change(status):
        height_str = f"{status.height_cm:.1f} cm ({status.height_cm / 2.54:.1f} in)" if status.height_cm else "unknown"
        print(f"\r  [update] state={status.state.value}  position={status.position.value}  height={height_str}", flush=True)

    desk.on_state_change(on_state_change)

    print(f"Connecting (sitstand_path={sitstand_path})...")
    connected = await desk.connect()
    if not connected:
        print("Error: Could not connect to desk")
        return 1

    status = desk.get_status()
    if status.state == DeskState.CONNECTED and desk._desk_control is None:
        print("Running in SIMULATION mode (no desk_control module found)")
    else:
        print("Connected to desk via BLE")

    print()
    print("Commands:")
    print("  s - Sit (move desk down)")
    print("  t - Stand (move desk up)")
    print("  h - Print current height/status")
    print("  q - Quit")
    print("=" * 40)

    try:
        while True:
            try:
                cmd = (await read_command()).strip().lower()
            except EOFError:
                break

            if cmd == "q":
                print("Quitting...")
                break
            elif cmd == "s":
                print("Moving to SIT position...")
                ok = await desk.sit()
                print(f"  sit() returned {'OK' if ok else 'FAILED'}")
            elif cmd == "t":
                print("Moving to STAND position...")
                ok = await desk.stand()
                print(f"  stand() returned {'OK' if ok else 'FAILED'}")
            elif cmd == "h":
                status = desk.get_status()
                height = await desk.get_height()
                height_str = f"{height:.1f} cm ({height / 2.54:.1f} in)" if height else "unknown"
                print(f"  State:    {status.state.value}")
                print(f"  Position: {status.position.value}")
                print(f"  Height:   {height_str}")
                if status.last_command:
                    print(f"  Last cmd: {status.last_command}")
                if status.error_message:
                    print(f"  Error:    {status.error_message}")
            elif cmd == "":
                continue
            else:
                print(f"  Unknown command: '{cmd}' (use s/t/h/q)")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        await desk.disconnect()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
