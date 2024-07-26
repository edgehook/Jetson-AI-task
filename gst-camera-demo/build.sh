#!/bin/bash -x

g++ -Wall -std=c++11 gst-demo.cpp -o gst-demo $(pkg-config --cflags --libs gstreamer-app-1.0) -ldl

