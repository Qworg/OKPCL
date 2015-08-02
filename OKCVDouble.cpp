/*
 * This file contains code that is part of the OpenKinect Project.
 * http://www.openkinect.org
 *
 * Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 * Additional code is copyright (c) 2011 Jeff Kramer (jeffkramr@gmail.com).
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 * 
 * 
 */

#include <iostream>
#include <libfreenect.hpp>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <ctime>
#include <boost/thread/thread.hpp>
//OPENCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std; 

///OKCVMutex Class
class OKCVMutex {
public:
	OKCVMutex() {
		pthread_mutex_init( &m_mutex, NULL );
	}
	void lock() {
		pthread_mutex_lock( &m_mutex );
	}
	void unlock() {
		pthread_mutex_unlock( &m_mutex );
	}

	class ScopedLock
	{
		OKCVMutex & _mutex;
	public:
		ScopedLock(OKCVMutex & mutex)
			: _mutex(mutex)
		{
			_mutex.lock();
		}
		~ScopedLock()
		{
			_mutex.unlock();
		}
	};
private:
	pthread_mutex_t m_mutex;
};


///Kinect Hardware Connection Class
/* thanks to Yoda---- from IRC */
class MyFreenectDevice : public Freenect::FreenectDevice {
public:
	MyFreenectDevice(freenect_context *_ctx, int _index)
		: Freenect::FreenectDevice(_ctx, _index), depth(freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes),m_buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes), m_new_rgb_frame(false), m_new_depth_frame(false)
	{
		
	}
	//~MyFreenectDevice(){}
	// Do not call directly even in child
	void VideoCallback(void* _rgb, uint32_t timestamp) {
		OKCVMutex::ScopedLock lock(m_rgb_mutex);
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		std::copy(rgb, rgb+getVideoBufferSize(), m_buffer_video.begin());
		m_new_rgb_frame = true;
	};
	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {
		OKCVMutex::ScopedLock lock(m_depth_mutex);
		depth.clear();
		uint16_t* call_depth = static_cast<uint16_t*>(_depth);
		for (size_t i = 0; i < 640*480 ; i++) {
			depth.push_back(call_depth[i]);
		}
		m_new_depth_frame = true;
	}
	bool getRGB(std::vector<uint8_t> &buffer) {
		//printf("Getting RGB!\n");
		OKCVMutex::ScopedLock lock(m_rgb_mutex);
		if (!m_new_rgb_frame) {
			//printf("No new RGB Frame.\n");
			return false;
		}
		buffer.swap(m_buffer_video);
		m_new_rgb_frame = false;
		return true;
	}

	bool getDepth(std::vector<uint16_t> &buffer) {
		OKCVMutex::ScopedLock lock(m_depth_mutex);
		if (!m_new_depth_frame)
			return false;
		buffer.swap(depth);
		m_new_depth_frame = false;
		return true;
	}

private:
	std::vector<uint16_t> depth;
	std::vector<uint8_t> m_buffer_video;
	OKCVMutex m_rgb_mutex;
	OKCVMutex m_depth_mutex;
	bool m_new_rgb_frame;
	bool m_new_depth_frame;
};


///Start the PCL/OK Bridging

//OK
Freenect::Freenect freenect;
MyFreenectDevice* device;
MyFreenectDevice* devicetwo;
freenect_video_format requested_format(FREENECT_VIDEO_RGB);
double freenect_angle(0);
int got_frames(0),window(0);
int g_argc;
char **g_argv;
int user_data = 0;


//OpenCV



///Keyboard Event Tracking
/*void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{

}*/


// --------------
// -----Main-----
// --------------
int main (int argc, char** argv)
{
	//More Kinect Setup
	static std::vector<uint16_t> kdepth(640*480);
	static std::vector<uint8_t> krgb(640*480*4);
	static std::vector<uint16_t> tdepth(640*480);
	static std::vector<uint8_t> trgb(640*480*4);
	
	// Create and setup OpenCV
    Mat mRGB (480, 640, CV_8UC3);
    Mat mDepth (480, 640, CV_16UC1);
    Mat tRGB (480, 640, CV_8UC3);
    Mat tDepth (480, 640, CV_16UC1);

    namedWindow("Kinect1", CV_WINDOW_AUTOSIZE);
    //namedWindow("Kinect2", CV_WINDOW_AUTOSIZE);	

  	printf("Create the devices.\n");
  	device = &freenect.createDevice<MyFreenectDevice>(0);
  	devicetwo = &freenect.createDevice<MyFreenectDevice>(1);
	device->startVideo();
	device->startDepth();
	boost::this_thread::sleep (boost::posix_time::seconds (1));
	devicetwo->startVideo();
	devicetwo->startDepth();
	boost::this_thread::sleep (boost::posix_time::seconds (1));
	
	//Grab until clean returns
	int DepthCount = 0;
	while (DepthCount == 0) {
		device->updateState();
	 	device->getDepth(kdepth);
		device->getRGB(krgb);
		for (size_t i = 0;i < 480*640;i++)
			DepthCount+=kdepth[i];
	}

	//--------------------
  	// -----Main loop-----
	//--------------------
	double x = NULL;
	double y = NULL;
	double tx = NULL;
	double ty = NULL;
	int iRealDepth = 0;
	int iTDepth = 0;
	
	device->setVideoFormat(requested_format);
	devicetwo->setVideoFormat(requested_format);
	printf("Start the main loop.\n");
	
	
    while (1) {
   		device->updateState();
	 	device->getDepth(kdepth);
		device->getRGB(krgb);
	        		
		devicetwo->updateState();
	 	devicetwo->getDepth(tdepth);
		devicetwo->getRGB(trgb);
	
	    size_t i = 0;
   	 	size_t cinput = 0;
   	 	
    
	    for (size_t v=0 ; v<480 ; v++)
    	{
    		uint8_t* rowRPtr = mRGB.ptr<uint8_t>(v);
    		uint16_t* rowDPtr = mDepth.ptr<uint16_t>(v);
    		uint8_t* rowTRPtr = tRGB.ptr<uint8_t>(v);
    		uint16_t* rowTDPtr = tDepth.ptr<uint16_t>(v);
    		cinput = 0;
    		for ( size_t u=0 ; u<640 ; u++, i++, cinput++)
       		{
        		iRealDepth = kdepth[i];
        		iTDepth = tdepth[i];
       			freenect_camera_to_world(device->getDevicePtr(), u, v, iRealDepth, &x, &y);
				freenect_camera_to_world(devicetwo->getDevicePtr(), u, v, iTDepth, &tx, &ty);				
    			rowDPtr[u] = iRealDepth;
    			rowRPtr[(cinput*3)] = krgb[(i*3)+2];
    			rowRPtr[(cinput*3)+1] = krgb[(i*3)+1];
    			rowRPtr[(cinput*3)+2] = krgb[(i*3)];
    			rowTDPtr[u] = iTDepth;
    			rowTRPtr[(cinput*3)] = trgb[(i*3)+2];
    			rowTRPtr[(cinput*3)+1] = trgb[(i*3)+1];
    			rowTRPtr[(cinput*3)+2] = trgb[(i*3)];
    			//printf("RGB = %d,%d,%d\n", krgb[i*3],krgb[(i*3)+1],krgb[(i*3)+2]);
	       	}
		}
		
		//printf("Displaying mRGB of depth %d\n", mRGB.depth());
		
		Mat mGray;
		Mat mEdges;
 		Mat mDist;
 		//cvtColor( mRGB, mGray, CV_RGB2GRAY );
 		//Canny( mGray, mEdges, 50.0, 50.0, 3); 
 		//distanceTransform(mEdges, mDist, CV_DIST_L2, 5);
		/*Mat outImg = Mat(480, 1280, CV_8UC3);
		Rect roi (0,0,640,480);
		Mat right = outImg(roi);
		right = mRGB;
		Rect roi2 (640,0,640,480);
		Mat left = outImg(roi2);
		left = tRGB;*/
		imshow("Kinect1", mRGB);
		cvWaitKey(33);
		imshow("Kinect1", tRGB);
		cvWaitKey(33);
	}
	mRGB.release();
	mDepth.release();
	tRGB.release();
	tDepth.release();
    device->stopVideo();
	device->stopDepth();
	devicetwo->stopVideo();
	devicetwo->stopDepth();
	return 0;	

}
