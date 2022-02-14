.PHONY: build

CXXFLAGS_MINI:= -std=c++20 -Wfatal-errors -Wall -Wextra -shared -fPIC $(shell python3 -m pybind11 --includes) -Iducc/src -fvisibility=hidden
#CXXFLAGS:= -O3 -march=native -ffast-math $(CXXFLAGS_MINI)
CXXFLAGS:= $(CXXFLAGS_MINI)


OUT:= resolvelib$(shell python3-config --extension-suffix)

$(OUT) : resolvelib.cpp
	g++ $(CXXFLAGS) -o$(OUT) ducc/src/ducc0/infra/threading.cc $^

build : $(OUT)
