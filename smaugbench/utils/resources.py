"""
Detect the CPU and memory resources actually available to the current process.

On a plain machine this is simply the number of CPUs and the amount of RAM. On machines partitioned with
cgroups (e.g. systemd slices such as `ml-1slot-0.slice`, containers, SLURM jobs), the process may only be
allowed a fraction of the CPUs and of the RAM. Exceeding a cgroup `memory.high` limit does not crash the
process, it gets throttled and swapped, which makes everything extremely slow. This module reads these
limits so that the number of parallel workers can be chosen accordingly.

The limits can also be forced with the environment variables:
    SMAUGBENCH_MAX_CPUS     number of CPUs (e.g. 16)
    SMAUGBENCH_MAX_MEMORY   amount of memory in bytes, or with a suffix K, M, G, T (e.g. 120G)
"""

import os
import math
import argparse
from pathlib import Path

CGROUP_ROOT = Path('/sys/fs/cgroup')


def parse_size(value):
    """
    Convert a size like '120G', '512M' or '1073741824' to a number of bytes.
    """
    value = str(value).strip().upper().removesuffix('B').removesuffix('I')
    units = {'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4}
    if value and value[-1] in units:
        return int(float(value[:-1]) * units[value[-1]])
    return int(float(value))


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def _cgroup_dirs():
    """
    Return the cgroup directories of the current process, from the leaf up to the root. Limits set on any
    parent (e.g. the slice) also apply to the process, so all of them must be checked.
    Returns a dict {controller: [dirs]}; the controller is '' for cgroup v2.
    """
    content = _read('/proc/self/cgroup')
    if content is None:
        return {}
    dirs = {}
    for line in content.splitlines():
        _, controllers, rel_path = line.split(':', 2)
        for controller in controllers.split(','):
            if controller == '':
                # cgroup v2 (unified hierarchy)
                base = CGROUP_ROOT
            else:
                # cgroup v1: one hierarchy per controller (e.g. /sys/fs/cgroup/memory)
                base = next((p for p in (CGROUP_ROOT / controller, CGROUP_ROOT / controllers) if p.is_dir()), None)
                if base is None:
                    continue
            path = base / rel_path.lstrip('/')
            chain = [path, *path.parents]
            dirs[controller] = [d for d in chain if d == base or base in d.parents]
    return dirs


def _cgroup_memory_limit():
    """
    Smallest memory limit (bytes) set on the cgroups of the process, or None if there is no limit.
    """
    limits = []
    dirs = _cgroup_dirs()
    # cgroup v2: memory.high (throttling) and memory.max (OOM kill)
    for d in dirs.get('', []):
        for name in ('memory.high', 'memory.max'):
            value = _read(d / name)
            if value and value != 'max':
                limits.append(int(value))
    # cgroup v1: memory.limit_in_bytes (very large value when unlimited)
    for d in dirs.get('memory', []):
        value = _read(d / 'memory.limit_in_bytes')
        if value and int(value) < 2 ** 60:
            limits.append(int(value))
    return min(limits) if limits else None


def _cgroup_cpu_limit():
    """
    Smallest CPU quota (in number of CPUs, may be fractional) set on the cgroups of the process, or None.
    """
    limits = []
    dirs = _cgroup_dirs()
    # cgroup v2: cpu.max = "<quota> <period>" or "max <period>"
    for d in dirs.get('', []):
        value = _read(d / 'cpu.max')
        if value:
            quota, period = value.split()[:2]
            if quota != 'max':
                limits.append(int(quota) / int(period))
    # cgroup v1: cpu.cfs_quota_us = -1 when unlimited
    for d in dirs.get('cpu', []):
        quota, period = _read(d / 'cpu.cfs_quota_us'), _read(d / 'cpu.cfs_period_us')
        if quota and period and int(quota) > 0:
            limits.append(int(quota) / int(period))
    return min(limits) if limits else None


def available_cpus():
    """
    Number of CPUs this process can effectively use.
    """
    if os.environ.get('SMAUGBENCH_MAX_CPUS'):
        return max(1, int(os.environ['SMAUGBENCH_MAX_CPUS']))
    try:
        # CPUs the process is allowed to run on (Linux only)
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_limit()
    if quota is not None:
        cpus = min(cpus, math.ceil(quota))
    return max(1, cpus)


def available_memory():
    """
    Amount of memory (bytes) this process can use: the smallest between the free RAM of the machine and the
    cgroup memory limit.
    """
    if os.environ.get('SMAUGBENCH_MAX_MEMORY'):
        return parse_size(os.environ['SMAUGBENCH_MAX_MEMORY'])
    # MemAvailable on Linux, total physical memory otherwise (e.g. macOS)
    meminfo = _read('/proc/meminfo')
    memory = None
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith('MemAvailable:'):
                memory = int(line.split()[1]) * 1024
    if memory is None:
        memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    limit = _cgroup_memory_limit()
    if limit is not None:
        memory = min(memory, limit)
    return memory


def main():
    parser = argparse.ArgumentParser(
        description='Print the number of CPUs and the memory (GB) available to this process, taking cgroup '
                    'limits (e.g. systemd slices) into account. Output: "<cpus> <memory_gb>".'
    )
    parser.parse_args()
    print(available_cpus(), available_memory() // 1024 ** 3)


if __name__ == '__main__':
    main()
