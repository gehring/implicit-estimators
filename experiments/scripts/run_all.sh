#!/bin/bash

# assumed that this is being called from the "experiments folder
./scripts/run_explicit_chain.sh
./scripts/run_explicit_four_rooms.sh
./scripts/run_explicit_mountaincar.sh

./scripts/run_implicit_chain.sh
./scripts/run_implicit_four_rooms.sh
./scripts/run_implicit_mountaincar.sh