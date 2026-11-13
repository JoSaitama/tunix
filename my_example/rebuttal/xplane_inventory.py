"""Inventory XSpace planes without copying trace payloads or initializing JAX."""
import mmap
from pathlib import Path


def _varint(data, pos, end):
    value = 0
    for shift in range(0, 70, 7):
        if pos >= end:
            break
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Invalid or truncated XPlane varint")


def _messages(data, pos, end):
    """Yield length-delimited fields as offsets; skip scalar fields."""
    while pos < end:
        tag, pos = _varint(data, pos, end)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("Invalid XPlane field number")
        if wire == 2:
            size, pos = _varint(data, pos, end)
            stop = pos + size
            if stop > end:
                raise ValueError("Truncated XPlane message")
            yield number, pos, stop
            pos = stop
        elif wire == 0:
            _, pos = _varint(data, pos, end)
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        else:
            raise ValueError(f"Unsupported XPlane wire type {wire}")
    if pos != end:
        raise ValueError("Truncated XPlane scalar")


def inspect_xplane(path):
    # OpenXLA XSpace: planes=1, errors=2, warnings=3, hostnames=4.
    # XPlane: name=2, lines=3. XLine: events=4.
    # github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto
    path = Path(path)
    planes = []
    strings = {2: [], 3: [], 4: []}
    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        size = len(data)
        for field, start, end in _messages(data, 0, size):
            if field in strings:
                strings[field].append(data[start:end].decode("utf-8", errors="replace"))
            elif field == 1:
                plane = {"name": "", "lines": 0, "events": 0}
                for key, first, last in _messages(data, start, end):
                    if key == 2:
                        plane["name"] = data[first:last].decode("utf-8", errors="replace")
                    elif key == 3:
                        plane["lines"] += 1
                        plane["events"] += sum(n == 4 for n, _, _ in _messages(data, first, last))
                planes.append(plane)
    return {
        "trace_file": str(path), "size_bytes": size, "planes": planes,
        "errors": strings[2], "warnings": strings[3], "hostnames": strings[4],
        "has_tpu_device_events": any(p["name"].startswith("/device:TPU:") and p["events"] > 0
                                     for p in planes),
    }
