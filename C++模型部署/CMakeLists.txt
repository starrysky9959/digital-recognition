cmake_minimum_required(VERSION 3.7)
project(deploy)

set(CMAKE_CXX_STANDARD 11)

# 指定文件夹位置
set(OPENCV_DIR /home/luzhan/软件/opencv-3.4.3)
#set(Torch_DIR /home/luzhan/软件/libtorch_cpu/share/cmake/Torch)
set(Torch_DIR /home/luzhan/软件/libtorch_gpu/share/cmake/Torch)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# 自动查找包
find_package(Torch REQUIRED)
find_package(OpenCV 3.4.3 REQUIRED)

# 添加源程序
add_executable(deploy
        main.cpp
        )

# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS} ./include)

# 加入库文件位置
target_link_libraries(deploy
        ${OpenCV_LIBS}
        -pthread
        -lMVSDK
        /lib/libMVSDK.so
        )
target_link_libraries(deploy
        ${TORCH_LIBRARIES}
        )