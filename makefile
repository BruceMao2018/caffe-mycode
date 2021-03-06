OBJS = atrinet_classfication.cpp
APP_NAME = atr_cpu
CC = g++
CFLAGS = -d -g
FLAGS =

ROOT_PATH=/home/bruce/local_install
LIBRARY = $(ROOT_PATH)/caffe/.build_release/lib/ -L $(ROOT_PATH)/caffe/build/lib/ -L $(ROOT_PATH)/lib/ -lcaffe -lglog -lprotobuf -lboost_system -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lstdc++ -lopencv_core

INCLUDE = /usr/local/cuda/include/ -I /usr/local/include/ -I $(ROOT_PATH)/include/ -I $(ROOT_PATH)/caffe/include -I $(ROOT_PATH)/caffe/.build_release/src/

default:$(OBJS)
	$(CC) -o $(APP_NAME) $(OBJS) $(FLAGS) -I$(INCLUDE) -L$(LIBRARY)
clean:
	rm $(APP_NAME)
