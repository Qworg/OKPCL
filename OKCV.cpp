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
//OPENCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

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
freenect_video_format requested_format(FREENECT_VIDEO_RGB);
double freenect_angle(0);
int got_frames(0),window(0);
int g_argc;
char **g_argv;
int user_data = 0;


//OpenCV
Mat mCorners;
Mat mOut;
Mat mGray;
Size boardSize(10,7); //interior number of corners
Size imageSize;
float squareSize = 0.023; //23 mm
Mat cameraMatrix, distCoeffs;
vector<vector<Point2f> > imagePoints;
vector<Point2f> pointbuf;
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


// --------------
// -----Main-----
// --------------
int main (int argc, char** argv)
{
	//More Kinect Setup
	static std::vector<uint16_t> kdepth(640*480);
	static std::vector<uint8_t> krgb(640*480*4);
	
	// Create and setup OpenCV
    Mat mRGB (640, 480, CV_8UC3);
    Mat mDepth (640, 480, CV_16UC1);    
    imageSize = mRGB.size();
    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
    //cvNamedWindow("Depth", CV_WINDOW_AUTOSIZE);	

  	printf("Create the device.\n");
  	device = &freenect.createDevice<MyFreenectDevice>(0);
	device->startVideo();
	device->startDepth();
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
	int iRealDepth = 0;
	int iTDepth = 0;
	
	device->setVideoFormat(requested_format);
	printf("Start the main loop.\n");
	
	int state = 0;
	
    while (1) {
   		device->updateState();
	 	device->getDepth(kdepth);
		device->getRGB(krgb);
	    
	    size_t i = 0;
   	 	size_t cinput = 0;
   	 	
    
	    for (size_t v=0 ; v<480 ; v++)
    	{
    		uint8_t* rowRPtr = mRGB.ptr<uint8_t>(v);
    		uint16_t* rowDPtr = mDepth.ptr<uint16_t>(v);
    		cinput = 0;
    		for ( size_t u=0 ; u<640 ; u++, i++, cinput++)
       		{
       			//pcl::PointXYZRGB result;
	       		iRealDepth = kdepth[i];
    			freenect_camera_to_world(device->getDevice(), u, v, iRealDepth, &x, &y);
    			rowDPtr[u] = iRealDepth;
    			rowRPtr[(cinput*3)] = krgb[(i*3)+2];
    			rowRPtr[(cinput*3)+1] = krgb[(i*3)+1];
    			rowRPtr[(cinput*3)+2] = krgb[(i*3)];
    			//printf("RGB = %d,%d,%d\n", krgb[i*3],krgb[(i*3)+1],krgb[(i*3)+2]);
	       	}
		}
		
		//printf("Displaying mRGB of depth %d\n", mRGB.depth());
		
		cvtColor( mRGB, mGray, CV_RGB2GRAY );
		
		
		if (state == 0) {
			bool found = false;
			//CALIB_CB_FAST_CHECK saves a lot of time on images
			//that do not contain any chessboard corners
			found = findChessboardCorners( mGray, boardSize, pointbuf,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
                    
			if(found)
			{
				cornerSubPix( mGray, pointbuf, Size(11,11),
            	Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	            imagePoints.push_back(pointbuf);
    	        drawChessboardCorners( mRGB, boardSize, Mat(pointbuf), found );
        	}
        	imshow("Image", mRGB);
        	
		} else {
			//Calibrated!
			remap(mRGB, mCalib, map1, map2, INTER_LINEAR);
			imshow("Image", mCalib);
            //int c = 0xff & waitKey();
            //if( (c & 255) == 27 || c == 'q' || c == 'Q' )
            //    break;
		}
		
		
		//imshow("Depth", mDepth);
		
		
		if (imagePoints.size() >= 100) {
			cout << "Calculating Distortion and Camera Matrix";
			state = 1;
			double totalAvgErr = 0;
    
   			 bool ok = runCalibration(imagePoints, imageSize, boardSize, squareSize, aspectRatio, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, totalAvgErr);
		    printf("%s. avg reprojection error = %.2f\n", ok ? "Calibration succeeded" : "Calibration failed", totalAvgErr);
    
		    cout << "Camera Matrix: " << cameraMatrix << endl;
		    cout << "Dist Coeffs: " << distCoeffs << endl;
		    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                imageSize, CV_16SC2, map1, map2);
		}
		
		cvWaitKey(66);
		
	}
	mRGB.release();
	mDepth.release();
    device->stopVideo();
	device->stopDepth();	
	return 0;	

}