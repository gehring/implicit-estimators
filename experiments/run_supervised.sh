#!/bin/bash

trap "exit 1" SIGSEGV

python supervised.py "$@"
exit