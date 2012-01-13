/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
 * for details.
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
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
//OPENCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/eigen.hpp"
//PCL Includes
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
		//printf("Getting RGB!\n");
		Mutex::ScopedLock lock(m_rgb_mutex);
		if (!m_new_rgb_frame) {
			//printf("No new RGB Frame.\n");
			return false;
		}
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
pcl::PointCloud<pcl::PointXYZRGB>::Ptr bgcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
float resolution = 50.0;
// Instantiate octree-based point cloud change detection class
pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZRGB> octree (resolution);

bool BackgroundSub = false;
bool hasBackground = false;
bool Voxelize = false;
unsigned int voxelsize = 10; //in mm
unsigned int cloud_id = 0;


//OpenCV
Mat mGray;
Size boardSize(10,7); //interior number of corners
Size imageSize;
float squareSize = 0.023; //23 mm
Mat camera1Matrix, dist1Coeffs, camera2Matrix, dist2Coeffs;
Mat R, T, E, F;
vector<vector<Point2f> > image1Points;
vector<vector<Point2f> > image2Points;
vector<Point2f> pointbuf, pointbuf2;
float aspectRatio = 1.0f;
vector<Mat> rvecs, tvecs;
vector<float> reprojErrs;
Mat map1, map2;
Mat mCalib;

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.resize(0);
    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
}

static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());
    
    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }
    
    return std::sqrt(totalErr/totalPoints);
}

static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize,
                    float squareSize, float aspectRatio,
                    Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    
    distCoeffs = Mat::zeros(8, 1, CV_64F);
    
    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0]);
    objectPoints.resize(imagePoints.size(),objectPoints[0]);
    
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                    distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
                    ///*|CV_CALIB_FIX_K3*/|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);
    
    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
    
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

static bool runStereo ( vector<vector<Point2f> > image1Pt,
					vector<vector<Point2f> > image2Pt, Size imageSize, Size boardSize, float squareSize, float aspectRatio, Mat c1Matrix, Mat c2Matrix, Mat d1Coeffs, Mat d2Coeffs, Mat& R, Mat& T, Mat& E, Mat& F, double& totalAvgErr)
{
	
	vector<vector<Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0]);
	objectPoints.resize(image1Pt.size(),objectPoints[0]);
    
    double rms = stereoCalibrate(objectPoints, image1Pt, image2Pt,
                    c1Matrix, d1Coeffs,
                    c2Matrix, d2Coeffs,
                    imageSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                    CV_CALIB_FIX_INTRINSIC +
                    CV_CALIB_FIX_ASPECT_RATIO +
                    CV_CALIB_ZERO_TANGENT_DIST +
                    CV_CALIB_SAME_FOCAL_LENGTH);
    cout << "Stereo Done with RMS Error=" << rms << endl;
    
    // CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(int i = 0; i < image1Pt.size(); i++ )
    {
        int npt = (int)image1Pt[i].size();
        Mat imgpt[2];
        imgpt[0] = Mat(image1Pt[i]);
	    undistortPoints(imgpt[0], imgpt[0], c1Matrix, d1Coeffs, Mat(), c1Matrix);
        computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);
        imgpt[1] = Mat(image2Pt[i]);
	    undistortPoints(imgpt[1], imgpt[1], c2Matrix, d2Coeffs, Mat(), c2Matrix);
        computeCorrespondEpilines(imgpt[1], 2, F, lines[1]);
        
        for(int j = 0; j < npt; j++ )
        {
            double errij = fabs(image1Pt[i][j].x*lines[1][j][0] +
                                image1Pt[i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(image2Pt[i][j].x*lines[0][j][0] +
                                image2Pt[i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    totalAvgErr = err/npoints;
    
    return true;
	
}


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

  if (event.getKeySym () == "b" && event.keyDown ())
  {
  	std::cout << "b was pressed" << std::endl;
  	if (BackgroundSub == false) 
  	{
  		//Start background subtraction
  		if (hasBackground == false) 
  		{
  			//Copy over the current cloud as a BG cloud.
  			pcl::copyPointCloud(*cloud, *bgcloud);
  			hasBackground = true;
  		}
  		BackgroundSub = true;
  	}
  	else 
  	{
  		//Stop Background Subtraction
  		BackgroundSub = false;
  	}
  }
  
  if (event.getKeySym () == "v" && event.keyDown ())
  {
  	std::cout << "v was pressed" << std::endl;
  	Voxelize = !Voxelize;
  }

}

// --------------
// -----Main-----
// --------------
int main (int argc, char** argv)
{
	int State = 0; //0 = Calib 1, 1 = Calib 2, 2 = Calib Stereo, 3 = PCL convert
	bool load = false;
	// create an empty vector of strings
    vector<string> args;
    // copy program arguments into vector
    if (argc > 1) {
	    for (int i=1;i<argc;i++) 
    	    args.push_back(argv[i]);
    	//LOAD THINGS!
    	load = true;
    	FileStorage fs(args[0], FileStorage::READ);
    	fs["camera1Matrix"] >> camera1Matrix;
    	fs["dist1Coeffs"] >> dist1Coeffs;
    	fs["camera2Matrix"] >> camera2Matrix;
    	fs["dist2Coeffs"] >> dist2Coeffs;
    	fs["R"] >> R;
    	fs["T"] >> T;    
    	fs.release();
    	State = 3;
    }
	
	
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
    imageSize = mRGB.size();

	if (!load)
	    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
    
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
	//Create the Goal Transform for PCL
  	Eigen::Vector3f PCTrans;
  	Eigen::Quaternionf PCRot;
  	
  	// Create the viewer
	boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer;

	//Loaded?
	if (load) {
		Eigen::Matrix3f eRot;
		cv2eigen(R,eRot);
		PCRot = Eigen::Quaternionf(eRot);
		cv2eigen(T,PCTrans);
		PCTrans*=1000; //meters to mm				

		//Open PCL section
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("3D Viewer"));
		boost::this_thread::sleep (boost::posix_time::seconds (1));

		viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
		viewer->setBackgroundColor (255, 255, 255);
		viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "Kinect Cloud");
		viewer->addPointCloud<pcl::PointXYZRGB> (cloud2, "Second Cloud");
	  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Kinect Cloud");
	  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Second Cloud");
	  	viewer->addCoordinateSystem (1.0);
		viewer->initCameraParameters ();
		printf("Viewer Built!  Displaying 3D Point Clouds\n");
	}

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
       			freenect_camera_to_world(device->getDevice(), u, v, iRealDepth, &x, &y);
				freenect_camera_to_world(devicetwo->getDevice(), u, v, iTDepth, &tx, &ty);				
    			rowDPtr[u] = iRealDepth;
	    		rowRPtr[(cinput*3)] = krgb[(i*3)+2];
    			rowRPtr[(cinput*3)+1] = krgb[(i*3)+1];
    			rowRPtr[(cinput*3)+2] = krgb[(i*3)];    		
    			rowTDPtr[u] = iTDepth;
   				rowTRPtr[(cinput*3)] = trgb[(i*3)+2];
    			rowTRPtr[(cinput*3)+1] = trgb[(i*3)+1];
   				rowTRPtr[(cinput*3)+2] = trgb[(i*3)];
    				
   				cloud->points[i].x  = x;//1000.0;
				cloud->points[i].y  = y;//1000.0;
        	    cloud->points[i].z = iRealDepth;//1000.0;
   	        	cloud->points[i].r = krgb[i*3];
        	    cloud->points[i].g = krgb[(i*3)+1];
   		        cloud->points[i].b = krgb[(i*3)+2];  				
  				
				cloud2->points[i].x  = tx;//1000.0;
				cloud2->points[i].y  = ty;//1000.0;
   	    	    cloud2->points[i].z = iTDepth;//1000.0;
       	    	cloud2->points[i].r = trgb[i*3];
        	    cloud2->points[i].g = trgb[(i*3)+1];
   	        	cloud2->points[i].b = trgb[(i*3)+2];
	       	}
		}
		
		//printf("Displaying mRGB of depth %d\n", mRGB.depth());
		
		if (!load && State < 3) {
			if (State == 0) {
				cvtColor( mRGB, mGray, CV_RGB2GRAY );
				bool found = false;
				//CALIB_CB_FAST_CHECK saves a lot of time on images
				//that do not contain any chessboard corners
				found = findChessboardCorners( mGray, boardSize, pointbuf,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
                    
				if(found)
				{
					cornerSubPix( mGray, pointbuf, Size(11,11),
            		Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	            	image1Points.push_back(pointbuf);
	    	        drawChessboardCorners( mRGB, boardSize, Mat(pointbuf), found );
    	    	}
        		imshow("Image", mRGB);
			}
			else if (State == 1) {
				cvtColor( tRGB, mGray, CV_RGB2GRAY );
				bool found = false;
				//CALIB_CB_FAST_CHECK saves a lot of time on images
				//that do not contain any chessboard corners
				found = findChessboardCorners( mGray, boardSize, pointbuf,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
                    
				if(found)
				{
					cornerSubPix( mGray, pointbuf, Size(11,11),
            		Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	            	image2Points.push_back(pointbuf);
	    	        drawChessboardCorners( tRGB, boardSize, Mat(pointbuf), found );
    	    	}
        		imshow("Image", tRGB);
        	}
        	else if (State == 2) {
        		//Stereo Calibration
        		cvtColor( mRGB, mGray, CV_RGB2GRAY );
				bool found1 = false;
				//CALIB_CB_FAST_CHECK saves a lot of time on images
				//that do not contain any chessboard corners
				found1 = findChessboardCorners( mGray, boardSize, pointbuf,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
                    
				if(found1)
				{
					cornerSubPix( mGray, pointbuf, Size(11,11),
            		Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	    	        drawChessboardCorners( mRGB, boardSize, Mat(pointbuf), found1);
    	    	}
    	    	imshow("Image", mRGB);
        		cvtColor( tRGB, mGray, CV_RGB2GRAY );
				bool found2 = false;
				//CALIB_CB_FAST_CHECK saves a lot of time on images
				//that do not contain any chessboard corners
				found2 = findChessboardCorners( mGray, boardSize, pointbuf2,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
                    
				if(found2)
				{
					cornerSubPix( mGray, pointbuf2, Size(11,11),
            		Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	    	        drawChessboardCorners( tRGB, boardSize, Mat(pointbuf2), found2 );
	    	        imshow("Image", tRGB);
    	    	}
    	    	
    	    	if (found1 && found2)
    	    	{
    	    		image1Points.push_back(pointbuf);
    	    		image2Points.push_back(pointbuf2);
    	    		printf("%d\n", image1Points.size());
    	    	}
        		
        	}

			if (State == 0 && image1Points.size() >= 100) {
				cout << "Calculating Distortion and Camera Matrix for Image 1" << endl;
				State = 1;
				double totalAvgErr = 0;
    
   				bool ok = runCalibration(image1Points, imageSize, boardSize, squareSize, aspectRatio, camera1Matrix, dist1Coeffs, rvecs, tvecs, reprojErrs, totalAvgErr);
			    printf("%s. avg reprojection error = %.2f\n", ok ? "Calibration succeeded" : "Calibration failed", totalAvgErr);
    
			    cout << "Camera Matrix: " << camera1Matrix << endl;
		    	cout << "Dist Coeffs: " << dist1Coeffs << endl;
		    }
		    
		    if (State == 1 && image2Points.size() >= 100) {
				cout << "Calculating Distortion and Camera Matrix for Image 2" << endl;
				State = 2;
				double totalAvgErr = 0;
    
   				bool ok = runCalibration(image2Points, imageSize, boardSize, squareSize, aspectRatio, camera2Matrix, dist2Coeffs, rvecs, tvecs, reprojErrs, totalAvgErr);
			    printf("%s. avg reprojection error = %.2f\n", ok ? "Calibration succeeded" : "Calibration failed", totalAvgErr);
    
			    cout << "Camera Matrix: " << camera2Matrix << endl;
		    	cout << "Dist Coeffs: " << dist2Coeffs << endl;
		    	image1Points.clear();
		    	image2Points.clear();
		    }
		    
		    if (State == 2 && image1Points.size() >= 100 && image2Points.size() >= 100)
		    {
		    	State = 3;
		    	double totalAvgErr = 0;
		    	
		    	
		    	bool ok = runStereo ( image1Points,
				image2Points, imageSize, boardSize, squareSize, aspectRatio, camera1Matrix, camera2Matrix, dist1Coeffs, dist2Coeffs, R, T, E, F, totalAvgErr);
		    	cout << "Stereo Avg Repro Err" << totalAvgErr << endl;	
		    	cout << "Rotation: " << R << endl;
		    	cout << "Translation: " << T << endl;
		    	
		    	Eigen::Matrix3f eRot;
		    	cv2eigen(R,eRot);
		    	PCRot = Eigen::Quaternionf(eRot);
		    	cv2eigen(T,PCTrans);
		    	PCTrans*=1000; //meters to mm
		    	
		    	//Store the Data from this calib
		    	printf("Writing calib out to calib.yaml.");
		    	FileStorage fs("calib.yaml", FileStorage::WRITE);
		    	fs << "camera1Matrix" << camera1Matrix;
    			fs << "dist1Coeffs"  << dist1Coeffs;
    			fs << "camera2Matrix" << camera2Matrix;
    			fs << "dist2Coeffs" << dist2Coeffs;
    			fs << "R" << R;
    			fs << "T" << T;    
    			fs.release();
		    	
		    	//Close out OpenCV section
		    	destroyAllWindows();
				boost::this_thread::sleep (boost::posix_time::seconds (1));

				
				//Open PCL section
				viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("3D Viewer"));
				boost::this_thread::sleep (boost::posix_time::seconds (1));

				viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
				viewer->setBackgroundColor (255, 255, 255);
				viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "Kinect Cloud");
  				viewer->addPointCloud<pcl::PointXYZRGB> (cloud2, "Second Cloud");
			  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Kinect Cloud");
			  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Second Cloud");
			  	viewer->addCoordinateSystem (1.0);
				viewer->initCameraParameters ();
				printf("Viewer Built!  Displaying 3D Point Clouds\n");
				continue;
		    }
		    
		    
			cvWaitKey(66);
		}
		else
		{
			if (!viewer->wasStopped ()) 
			{
				pcl::transformPointCloud (*cloud, *cloud, PCTrans, PCRot);
		    	viewer->updatePointCloud (cloud, "Kinect Cloud");
	    		viewer->updatePointCloud (cloud2, "Second Cloud");
	    		viewer->spinOnce ();
	    	}
	    	else
	    	{
	    		break;
	    	}
	    }	
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