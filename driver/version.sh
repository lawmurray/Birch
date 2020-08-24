git describe --tags --dirty --always | sed -E 's/v([0-9]+)-([0-9]+)-g[a-f0-9]+/\1.\2/' || echo 0
