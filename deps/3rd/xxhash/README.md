xxHash 0.8.1, upstream `xxhash.h` under its embedded BSD 2-Clause license.

Source: https://github.com/Cyan4973/xxHash/tree/v0.8.1

Only the upstream header is vendored. Zephyr uses its private inline XXH64
implementation with XXH3 disabled; there is no runtime or system-library dependency.
