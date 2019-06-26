OBJS = DnnCaffe.cpp
APP_NAME = bubble_dnn
CC = g++
CFLAGS = -d -g
FLAGS =

ROOT_PATH=/home/bruce/local_install

LIBRARY_BASE = $(ROOT_PATH)/caffe/.build_release/lib/ -L $(ROOT_PATH)/caffe/build/lib/ -L $(ROOT_PATH)/lib/ -lcaffe -lglog -lprotobuf -lboost_system -lstdc++

#-L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_dnn -lopencv_ml -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev

OPENCV_LIBRARY = `pkg-config opencv --cflags --libs`

LIBRARY = $(LIBRARY_BASE) $(OPENCV_LIBRARY)

INCLUDE = /usr/local/cuda/include/ -I /usr/local/include/ -I $(ROOT_PATH)/include/ -I $(ROOT_PATH)/caffe/include -I $(ROOT_PATH)/caffe/.build_release/src/ -I/usr/local/include/opencv

default:$(OBJS)
	$(CC) -o $(APP_NAME) $(OBJS) $(FLAGS) -I$(INCLUDE) -L$(LIBRARY)
clean:
	rm $(APP_NAME)

#above same as g++ -o bubble_dnn DnnCaffe.cpp `pkg-config opencv --cflags --libs`
