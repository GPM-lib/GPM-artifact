CXXFLAGS=
LDFLAGS=
#ASAN=1

ifeq ($(ASAN),1)
CFLAGS += -g -O0 -fsanitize=address
CXXFLAGS += -g -O0 -fsanitize=address
LDFLAGS += -fsanitize=address
endif

greader : greader.o
	g++ $(LDFLAGS) -o greader greader.o 
greader.o : greader.cpp greader.h
	g++ $(CXXFLAGS) -c greader.cpp 
clean :
	rm greader.o