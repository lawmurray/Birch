#!/bin/bash
set -eov pipefail

# online help
birch help
birch help init
birch help audit
birch help bootstrap
birch help configure
birch help build
birch help install
birch help uninstall
birch help dist
birch help docs
birch help clean
birch help help

# hello world package (but can't build without standard library)
mkdir hello
cd hello
birch init
birch audit
birch dist
birch docs
birch clean
cd ..
rm -rf hello
