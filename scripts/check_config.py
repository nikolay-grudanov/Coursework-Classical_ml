#!/usr/bin/env python3
"""Validate config paths referenced in configs/model_config.yaml

Exits with code 0 on success, 1 on failure. If PyYAML is missing, prints a message
and exits 0 (non-fatal, optional check).
"""

import os
import sys

try:
    import yaml
except Exception:
    print("PyYAML not installed; please install PyYAML to enable config checks")
    sys.exit(0)


def main():
    cfg_path = "configs/model_config.yaml"
    if not os.path.exists(cfg_path):
        print(f"Config file not found: {cfg_path}")
        return 1

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    keys = ["data_path", "processed_data_path"]
    errs = False
    for k in keys:
        v = cfg.get(k)
        if v is None:
            print(f"  MISSING key in config: {k}")
            errs = True
            continue

        # Resolve ${VAR:-default} pattern, $VAR and ~
        def resolve_path(val):
            if not isinstance(val, str):
                return val
            val = str(val)
            # handle ${VAR:-default}
            if val.startswith("${") and val.endswith("}") and ":-" in val:
                inner = val[2:-1]
                var, default = inner.split(":-", 1)
                return os.environ.get(var, default)
            return os.path.expanduser(os.path.expandvars(val))

        rv = resolve_path(v)
        if not os.path.exists(rv):
            print(f"  MISSING path for {k}: {rv}")
            errs = True
        else:
            print(f"  OK: {k} -> {rv}")

    if errs:
        print("Configuration validation failed")
        return 1

    print("Configuration validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
