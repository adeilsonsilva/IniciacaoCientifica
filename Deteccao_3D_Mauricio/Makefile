CC = g++

INCLUDE = -I /usr/include/libxml2/ `pkg-config --cflags opencv` -I /usr/local/include/libfreenect/
LDFLAGS = -lxml2 `pkg-config --libs opencv` -lfreenect_sync

FLAGS = -O3 -ffast-math

OBJ = detection.o main.o

PROG = a.out

all: $(PROG)

# Main program
$(PROG): $(OBJ)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJ) $(FLAGS) $(LDFLAGS)

# Multiple dependences
detection.o: kinect.hpp
main.o: kinect.hpp detection.hpp

# Default dependences
%.o: %.cpp %.hpp
%.o: %.cpp
	$(CC) -c $< $(INCLUDE)

# Clean
clean:
	rm -f *.o $(PROG)

