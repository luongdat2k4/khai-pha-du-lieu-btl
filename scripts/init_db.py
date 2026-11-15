"""Script to initialize runtime SQLite database and register existing artifacts."""

import os
import sys
import argparse

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.storage import init_db, DEFAULT_DB_PATH
from telco_churn.registry import (
    get_active_model_info,
    register_model_version,
    update_model_status,
)


def main():
    parser = argparse.ArgumentParser(description="Initialize Telco Churn SQLite runtime database.")
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Path to artifacts root directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Initializing SQLite database at: {args.db_path}")
    init_db(args.db_path)
    print("[+] Database tables and indexes created successfully.")

    # Register initial model if active_model.json exists
    active_path = os.path.join(args.artifacts_dir, "active_model.json")
    if os.path.exists(active_path):
        try:
            active_info = get_active_model_info(args.artifacts_dir)
            active_ver = active_info.get("active_version")
            model_dir = os.path.join(args.artifacts_dir, "models", active_ver)

            register_model_version(
                version=active_ver,
                run_id=active_info.get("metadata", {}).get("run_id", "initial_run"),
                artifact_path=model_dir,
                metrics=active_info.get("metadata", {}).get("test_metrics"),
                status="active",
                schema_version="1.0",
                db_path=args.db_path,
            )
            update_model_status(active_ver, "active", args.db_path)
            print(f"[+] Active model '{active_ver}' successfully registered into model_versions.")
        except Exception as e:
            print(f"[!] Warning: Could not register active model in DB: {e}")

    print("=" * 60)
    print("Database initialization complete.")


if __name__ == "__main__":
    main()
