### 1 采集摄像头保存至图片
gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=1 ! jpegenc ! filesink location=file.jpg

### 2 摄像头简单点亮
gst-launch-1.0 v4l2src device=/dev/video0  ! xvimagesink -ev
gst-launch-1.0 v4l2src device=/dev/video1  ! xvimagesink -ev

gst-launch-1.0 nvv4l2camerasrc device=/dev/video0 ! 'video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)2880, height=(int)1860, interlace-mode=progressive, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nv3dsink -e
gst-launch-1.0 nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)3840, height=(int)2160, interlace-mode=progressive, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nv3dsink -e

### 3 单路摄像头采集：
gst-launch-1.0 nvcompositor name=comp  sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 ! nv3dsink nvv4l2camerasrc device=/dev/video0 ! 'video/x-raw(memory:NVMM),format=(string)UYVY, width=2880, height=1860' ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)NV12" ! comp.
gst-launch-1.0 nvcompositor name=comp  sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 ! nv3dsink nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM),format=(string)UYVY, width=3840, height=2160' ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)NV12" ! comp.

两路同时采集：
gst-launch-1.0 nvcompositor name=comp  sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 sink_1::xpos=960 sink_1::ypos=0 sink_1::width=960 sink_1::height=540 ! nv3dsink nvv4l2camerasrc device=/dev/video0 ! 'video/x-raw(memory:NVMM),format=(string)UYVY, width=2880, height=1860' ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)NV12" ! comp. nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM),format=(string)UYVY, width=3840, height=2160' ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)NV12" ! comp.

### 4采集摄像头视频进行h.264编码, 保存至mp4文件
gst-launch-1.0 nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, format=(string)UYVY, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=test.mp4 -e

gst-launch-1.0 nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, format=(string)UYVY, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420' ! nvv4l2h264enc ! 'video/x-h264, stream-format=(string)byte-stream' ! h264parse ! qtmux ! filesink location=test.mp4 -e

### 5 *.mp4文件进行h.264解码并播放
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! queue ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nv3dsink -e

### 6 opencv调用处理gstreamer视频流，结果视频画面颜色异常（画面灰度）视频卡顿
gst-launch-1.0 v4l2src device=/dev/video1 ! 'video/x-raw, format=(string)UYVY, width=(int)3840, height=(int)2160, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw, format=(string)NV12' ! videoconvert ! appsink
gst-launch-1.0 nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)3840, height=(int)2160, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! videoconvert ! appsink

### 7 opencv调用处理gstreamer视频流，结果视频画面颜色正常视频卡顿，摄像头视频转换过中由于CPU参与导致卡顿
nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)3840, height=(int)2160, interlace-mode=progressive, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! 'video/x-raw, format=BGR' ! appsink

### 8 UDP流媒体传输
【服务器端】: v4l2摄像头捕获 + 视频编码 + 使用网络接收器的流传输
gst-launch-1.0 nvv4l2camerasrc device=/dev/video1 ! 'video/x-raw(memory:NVMM), format=UYVY, width=3840, height=2160, framerate=(fraction)30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvv4l2h264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! udpsink host=127.0.0.1 port=8001 sync=false -e

【客户端】: 网络源+视频解码+视频显示
gst-launch-1.0 udpsrc address=127.0.0.1 port=8001 caps='application/x-rtp, encoding-name=(string)H264, payload=(int)96' ! rtph264depay ! queue ! h264parse ! nvv4l2decoder ! nv3dsink -e

### 9 使用test-launch简单示例将摄像头（/dev/video1）通过gstreamer采集转成rtsp流
【rtsp服务端】:
./test-launch "nvv4l2camerasrc device="/dev/video1" ! video/x-raw(memory:NVMM),width=3840,height=2160 ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)NV12 ! nvv4l2h264enc ! rtph264pay name=pay0 pt=96"
【客户端】:
gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/test latency=500 ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nv3dsink -e

【示例代码：https://github.com/GStreamer/gst-rtsp-server.git】
gcc test-launch.c -o test-launch $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0)
