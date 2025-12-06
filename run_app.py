"""
Run the PyPath Shiny Dashboard.

Usage:
    python run_app.py
    
Or with custom port:
    python run_app.py --port 8080
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
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Import and run the app
    from app.app import app
    
    print(f"\n{'='*50}")
    print("  PyPath Dashboard")
    print(f"{'='*50}")
    print(f"\n  Starting server at http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
