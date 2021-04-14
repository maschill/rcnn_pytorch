# parent Docker image 
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/

RUN python --version

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install --trusted-host pypi.python.org -r requirements.txt

# build and install opencv4.3.0 from scratch
run apt-get update
run apt-get install -y wget apt-utils unzip apt-transport-https ca-certificates gnupg
run apt-get install -y software-properties-common
run wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
run apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
run apt-get update
run apt-get install -y cmake libtiff-dev libjpeg8-dev libpng-dev
run apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
run apt-get install -y libxine2-dev libv4l-dev libavresample-dev
run apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN DEBIAN_FRONTEND="noninteractive" TZ=EUROPE/BERLIN apt-get -y install tzdata
run apt-get install -y qt5-default gtk2.0 gtk3.0 libgtk2.0-dev libgtk-3-dev
run apt-get install -y libatlas-base-dev git
run apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
run apt-get install -y libvorbis-dev libxvidcore-dev
run apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
run apt-get install -y x264 x265 v4l-utils
run apt-get install -y libprotobuf-dev protobuf-compiler
run apt-get install -y libeigen3-dev

run wget --output-document cv.zip https://github.com/opencv/opencv/archive/master.zip
run wget --output-document contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip

ENV PATH=$PATH:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
RUN ldconfig

run unzip cv.zip
run unzip contrib.zip
workdir /app/opencv-master
run mkdir build
workdir /app/opencv-master/build
run cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_TBB=ON \
  -D WITH_V4L=ON \
  -D WITH_QT=ON \
  -D WITH_OPENGL=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUBLAS=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
  -D CUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/lubcudart.so \
  -D CUDA_VERBOSE_BUILD=ON \
  -D WITH_NVCUVID=OFF \
  -D BUILD_opencv_python=OFF \
  -D BUILD_opencv_python3=ON \
-D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
  -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
  -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
  -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-master/modules \
  -D OPENCV_ENABLE_NONFREE=ON \
  ..

run make -j4
run make install 
run ldconfig
run echo $(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
run echo $(python3 -c "import numpy; print(numpy.get_include())")
run ln -s /opt/conda/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38m-x86_64-linux-gnu.so /opt/conda/lib/python3.8/site-packages/cv2/python-3.8/cv2.so

WORKDIR /app
run conda list tensor