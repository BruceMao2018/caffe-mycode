OBJS = mat_at.cpp
APP_NAME = at
CC = g++
CFLAGS = -d -g
FLAGS =

LIBRARY = `pkg-config opencv --cflags --libs` -lbruceTime

INCLUDE = /usr/local/include/bruce/

default:$(OBJS)
	$(CC) -o $(APP_NAME) $(OBJS) $(FLAGS) -I$(INCLUDE) -L$(LIBRARY)
clean:
	rm $(APP_NAME)
