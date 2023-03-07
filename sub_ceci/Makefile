exec = myceci

cc = "$(shell which g++)"
flags = -I. -fopenmp -std=c++11 -O3 -g -Wall -march=native -pthread

objs = $(patsubst %.cpp, %.o, $(wildcard *.cpp)) \
		$(patsubst %.cpp, %.o, $(wildcard graph/*.cpp)) \
		$(patsubst %.cpp, %.o, $(wildcard utility/*.cpp))

deps = $(wildcard *.h) \
		$(wildcard graph/*.h) \
		$(wildcard configuration/*.h) \
		$(wildcard utility/*.h) \
		Makefile

%.o: %.cpp $(deps)
	$(cc) -c $< -o $@ ${flags}

$(exec):$(objs)
	$(cc) $(objs) -o $(exec) $(flags)

clean:
	rm -rf $(exec) $(objs)
