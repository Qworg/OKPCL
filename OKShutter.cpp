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
#include "pcl/common/common_headers.h"
#include "pcl/common/eigen.h"
#include "pcl/common/transforms.h"
#include "pcl/features/normal_3d.h"
#include "pcl/io/pcd_io.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/console/parse.h"
#include "pcl/point_types.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include "boost/lexical_cast.hpp"
#include "pcl/filters/voxel_grid.h"
#include "pcl/octree/octree.h"
//OPENCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"

using namespace cv;
using namespace std; 

///Mutex Class
class Mutex {
public:
	Mutex() {
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
		Mutex & _mutex;
	public:
		ScopedLock(Mutex & mutex)
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
		Mutex::ScopedLock lock(m_rgb_mutex);
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		std::copy(rgb, rgb+getVideoBufferSize(), m_buffer_video.begin());
		m_new_rgb_frame = true;
	};
	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {
		Mutex::ScopedLock lock(m_depth_mutex);
		depth.clear();
		uint16_t* call_depth = static_cast<uint16_t*>(_depth);
		for (size_t i = 0; i < 640*480 ; i++) {
			depth.push_back(call_depth[i]);
		}
		m_new_depth_frame = true;
	}
	bool getRGB(std::vector<uint8_t> &buffer) {
		Mutex::ScopedLock lock(m_rgb_mutex);
		if (!m_new_rgb_frame)
			return false;
		buffer.swap(m_buffer_video);
		m_new_rgb_frame = false;
		return true;
	}

	bool getDepth(std::vector<uint16_t> &buffer) {
		Mutex::ScopedLock lock(m_depth_mutex);
		if (!m_new_depth_frame)
			return false;
		buffer.swap(depth);
		m_new_depth_frame = false;
		return true;
	}

private:
	std::vector<uint16_t> depth;
	std::vector<uint8_t> m_buffer_video;
	Mutex m_rgb_mutex;
	Mutex m_depth_mutex;
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


//PCL
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1buf (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2buf (new pcl::PointCloud<pcl::PointXYZRGB>);

unsigned int cloud_id = 0;



///Keyboard Event Tracking
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  if (event.getKeySym () == "c" && event.keyDown ())
  {
    std::cout << "c was pressed => capturing a pointcloud" << std::endl;
    std::string filename = "KinectCap";
    filename.append(boost::lexical_cast<std::string>(cloud_id));
    filename.append(".pcd");
    pcl::io::savePCDFileASCII (filename, *cloud);
    cloud_id++;
  }
}


// --------------
// -----Main-----
// --------------
int main (int argc, char** argv)
{
	Mat camera1Matrix, dist1Coeffs, camera2Matrix, dist2Coeffs;
	Mat R, T;
	vector<string> args;
    // copy program arguments into vector
    if (argc > 1) {
	    for (int i=1;i<argc;i++) 
    	    args.push_back(argv[i]);
    	//LOAD THINGS!
    	FileStorage fs(args[0], FileStorage::READ);
    	fs["camera1Matrix"] >> camera1Matrix;
    	fs["dist1Coeffs"] >> dist1Coeffs;
    	fs["camera2Matrix"] >> camera2Matrix;
    	fs["dist2Coeffs"] >> dist2Coeffs;
    	fs["R"] >> R;
    	fs["T"] >> T;    
    	fs.release();
    } else {
    	printf("Wrong file to calib!");
    	return 0;
    }
	
	
	//Create the Goal Transform for PCL
  	Eigen::Vector3f PCTrans;
  	Eigen::Quaternionf PCRot;
	Eigen::Matrix3f eRot;
	cv2eigen(R,eRot);
	PCRot = Eigen::Quaternionf(eRot);
	cv2eigen(T,PCTrans);
	PCTrans*=1000; //meters to mm
	
	
	//More Kinect Setup
	static std::vector<uint16_t> mdepth(640*480);
	static std::vector<uint8_t> mrgb(640*480*4);
	static std::vector<uint16_t> tdepth(640*480);
	static std::vector<uint8_t> trgb(640*480*4);

	// Fill in the cloud data
	cloud->width    = 640;
	cloud->height   = 480;
	cloud->is_dense = false;
	cloud->points.resize (cloud->width * cloud->height);

	// Fill in the cloud data
	cloud2->width    = 640;
	cloud2->height   = 480;
	cloud2->is_dense = false;
	cloud2->points.resize (cloud2->width * cloud2->height);
	
	// Create and setup the viewer
	printf("Create the viewer.\n");
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "Kinect Cloud");
  	viewer->addPointCloud<pcl::PointXYZRGB> (cloud2, "Second Cloud");
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Kinect Cloud");
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Second Cloud");
  	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	
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
	 	device->getDepth(mdepth);
		device->getRGB(mrgb);
		for (size_t i = 0;i < 480*640;i++)
			DepthCount+=mdepth[i];
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
	float percentCoverage = 90.0;
	while (!viewer->wasStopped ())
	{
		device->updateState();
	 	device->getDepth(mdepth);
		device->getRGB(mrgb);
		
		devicetwo->updateState();
	 	devicetwo->getDepth(tdepth);
		devicetwo->getRGB(trgb);
	
	    size_t i = 0;
    	size_t cinput = 0;
    	int iMZeros = 0;
    	int iTZeros = 0;
    	for (size_t v=0 ; v<480 ; v++)
    	{
    		int mZeros = 0;
    		int tZeros = 0;
    		for ( size_t u=0 ; u<640 ; u++, i++)
        	{
        		iRealDepth = mdepth[i];
        		if (iRealDepth == 0) mZeros++;
        		iTDepth = tdepth[i];
        		if (iTDepth == 0) tZeros++;
				freenect_camera_to_world(device->getDevice(), u, v, iRealDepth, &x, &y);
				freenect_camera_to_world(devicetwo->getDevice(), u, v, iTDepth, &tx, &ty);
				
				cloud->points[i].x  = x;//1000.0;
				cloud->points[i].y  = y;//1000.0;
        	    cloud->points[i].z = iRealDepth;//1000.0;
            	cloud->points[i].r = mrgb[i*3];
	            cloud->points[i].g = mrgb[(i*3)+1];
    	        cloud->points[i].b = mrgb[(i*3)+2];  				
  				
  				cloud2->points[i].x  = tx;//1000.0;
				cloud2->points[i].y  = ty;//1000.0;
        	    cloud2->points[i].z = iTDepth;//1000.0;
            	cloud2->points[i].r = trgb[i*3];
	            cloud2->points[i].g = trgb[(i*3)+1];
    	        cloud2->points[i].b = trgb[(i*3)+2];
    	        
        	}
        	//printf("mZeros = %d for column %d\n", mZeros, v);
        	if (mZeros == 640) iMZeros++;
        	if (tZeros == 640) iTZeros++;
		}
		
		//printf("iMZeros = %d for image.  iTZeros = %d for image.\n", iMZeros, iTZeros);
		
		pcl::transformPointCloud (*cloud, *cloud, PCTrans, PCRot);
    	
    	//Only update the cloud if the other IR illuminator is 90% or better covered up.
    	if (iTZeros/480 >= percentCoverage/100.0) {
    		printf("Updating Cloud 1.\n");
    		viewer->updatePointCloud (cloud, "Kinect Cloud");
    	}
    	if (iMZeros/480 >= percentCoverage/100.0) {
    		printf("Updating Cloud 2.\n");
    		viewer->updatePointCloud (cloud2, "Second Cloud");
    	}
		
		viewer->spinOnce ();
	}
	device->stopVideo();
	device->stopDepth();
	devicetwo->stopVideo();
	devicetwo->stopDepth();
	return 0;
}