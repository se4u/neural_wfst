#!/usr/bin/env bash
# THEANO_FLAGS=device=cpu,cxx=clang++,gcc.cxxflags="-O1 -ffast-math -gsplit-dwarf $LDFLAGS"
# It takes 20 minutes to compile , even with clang. !!!
# THEANO_FLAGS=device=cpu,mode=FAST_COMPILE,gcc.cxxflags='-O1' - This only takes 45 second to compile
THEANO_FLAGS=device=cpu OMP_NUM_THREADS=${2-1} python -c "import transducer_score; print transducer_score.main($1)";
