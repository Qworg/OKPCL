cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_C_FLAGS "-Wall")

project(OKPCL)
find_package(PCL 1.6 REQUIRED)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKPCL OKPCL.cpp)

if(APPLE)
  target_link_libraries(OKPCL freenect ${PCL_LIBRARIES})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKPCL freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES})
endif()

project(OKPCLDouble)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKPCLDouble OKPCLDouble.cpp)

if(APPLE)
  target_link_libraries(OKPCLDouble freenect ${PCL_LIBRARIES})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKPCLDouble freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES})
endif()

project(OKCV)

find_package(OPENCV REQUIRED)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKCV OKCV.cpp)

if(APPLE)
  target_link_libraries(OKCV freenect ${PCL_LIBRARIES} ${OpenCV_LIBS})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKCV freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES} ${OpenCV_LIBS})
endif()

project(OKCVDouble)

find_package(OPENCV REQUIRED)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKCVDouble OKCVDouble.cpp)

if(APPLE)
  target_link_libraries(OKCVDouble freenect ${PCL_LIBRARIES} ${OpenCV_LIBS})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKCVDouble freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES} ${OpenCV_LIBS})
endif()

project(OKStereo)

find_package(OPENCV REQUIRED)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKStereo OKStereo.cpp)

if(APPLE)
  target_link_libraries(OKStereo freenect ${PCL_LIBRARIES} ${OpenCV_LIBS})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKStereo freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES} ${OpenCV_LIBS})
endif()

project(OKShutter)

find_package(OPENCV REQUIRED)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKShutter OKShutter.cpp)

if(APPLE)
  target_link_libraries(OKShutter freenect ${PCL_LIBRARIES} ${OpenCV_LIBS})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKShutter freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES} ${OpenCV_LIBS})
endif()

project(OKFlower)

if (WIN32)
  set(THREADS_USE_PTHREADS_WIN32 true)
  find_package(Threads REQUIRED)

  include_directories(${THREADS_PTHREADS_INCLUDE_DIR})
endif()

include_directories(. ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(OKFlower OKFlower.cpp AsyncSerial.cpp BufferedAsyncSerial.cpp)

if(APPLE)
  target_link_libraries(OKFlower freenect ${PCL_LIBRARIES})
else()
  find_package(Threads REQUIRED)
  include_directories(${USB_INCLUDE_DIRS})

  if (WIN32)
    set(MATH_LIB "")
  else(WIN32)
    set(MATH_LIB "m")
  endif()
  target_link_libraries(OKFlower freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${PCL_LIBRARIES})
endif()
