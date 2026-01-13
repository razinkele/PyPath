"""
Run the PyPath Shiny Dashboard.

This script provides a programmatic way to start the PyPath Shiny app.
For CLI-based startup, you can also use: shiny run app/app.py

Usage:
    python run_app.py

Or with custom port:
    python run_app.py --port 8080

Development with auto-reload (DO NOT use in production):
    python run_app.py --reload

Installation:
    pip install -e ".[web]"  # Install web dashboard dependencies
"""

import sys
import argparse
from pathlib import Path

# Add src to path for pypath imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run PyPath Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only, not for production)",
    )

    args = parser.parse_args()

    # Warn if reload is enabled
    if args.reload:
        print("\n⚠️  WARNING: Auto-reload is enabled. This is for DEVELOPMENT ONLY.")
        print("   Do not use --reload in production environments.\n")

    # Import and run the app
    from app.app import app

    print(f"\n{'='*50}")
    print("  PyPath Dashboard")
    print(f"{'='*50}")
    print(f"\n  Starting server at http://{args.host}:{args.port}")
    if args.reload:
        print("  Mode: Development (auto-reload enabled)")
    print("  Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
