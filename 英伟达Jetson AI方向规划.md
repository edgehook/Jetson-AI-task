## Jetson产品任务规划
| 任务号 |任务名|优先级| Status |
| -- | -- | -- | -- |
|1| Jetson AI产品BSP移植 | 紧急 | Doing | |
|2| 采集多路图像，视频处理 | 一般 | Todo | |
|3| Jetson支持多种AI框架 | 一般 | Todo | |
|4| AI网络视频录像机NVR | 一般 | Todo | |

---

### 1. Jetson AI产品BSP移植
- 基本设备树，内核移植     
- PCIE设备接口支持测试  
- 摄像头驱动支持&调试
- 板子基本外设接口调试

### 2. 采集多路图像，视频处理
- 多摄像头输入支持，支持方式采用[gst-launch-1.0](./gst-launch-1.0测试录制视频命令集合.md)指令和调用[gstreamer API](./gst-camera-demo/gst-demo.cpp)
- 通过[OpenCV](./opencv-sample-apps-modify/opencv_camera/opencv_gst_camera.cpp)，FFmpeg等图像和视频分析库，提供丰富的图像和视频处理功能
- NVIDIA提供的DeepStream SDK处理工具进行视频流的接收、解码、处理和输出，前期使用DeepStream提供的示例应用程序作为基础，进行进一步的开发和优化
- 根据具体的应用需求，配置视频处理流程，可能包括目标检测、跟踪、分类等任务

### 3. Jetson支持多种AI框架
- 支持热门AI框架，如[PyTorch](./使用PyTorch进行迁移模型学习.md)、TensorFlow等，为开发者提供灵活的开发环境
- 安装CUDA工具开发包和cuDNN库，以满足需要GPU加速的功能
- 开发和测试，确保它能够在GPU上运行
- 性能优化，使用 NVIDIA 提供的工具，如 TensorRT，对模型进行优化，以提高推理速度
- 链接：https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

### 4. AI网络视频录像机NVR
- 实时视频处理与分析：
	Jetson产品能够实时处理来自摄像头的视频流，利用其强大的计算能力，可以高效地运行视频分析算法。
- 目标检测与跟踪
	利用集成的AI算法模型，可以检测特定的目标，车辆等
- 链接：https://github.com/asmirnou/watsor 
	https://github.com/JoeTester1965/CudaCamz
