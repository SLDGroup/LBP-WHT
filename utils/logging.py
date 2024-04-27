import os
from pathlib import Path


def get_latest_log_version(path):
    max_version = -1
    for p in os.walk(path):
        p = p[0].split('/')[-1]
        if p.startswith('version_') and p[8:].isdigit():
            v = int(p[8:])
            if v > max_version:
                max_version = v
    return max_version + 1


def get_active_branch_name():
    head_dir = Path(".") / ".git" / "HEAD"
    if not head_dir.exists():
        return None
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]
