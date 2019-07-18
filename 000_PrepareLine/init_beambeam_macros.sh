#!/bin/bash
cd ..
git submodule init 000_PrepareLine/beambeam_macros
git submodule update 000_PrepareLine/beambeam_macros
cd 000_PrepareLine/beambeam_macros
make
cd ..
