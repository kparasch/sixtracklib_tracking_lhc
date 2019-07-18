#!/bin/bash
git submodule init beambeam_macros
git submodule update beambeam_macros
cd beambeam_macros
make
cd ..
