/** @mainpage Eye position tracker documentation
 @section intro Introducation
 This program computes an eyeball position in 3D given a calibrated camera image.
 @image html output_example.png A sample output

 @section build How to Compile
 A CMakeLists.txt is provided for building a solution via CMake.\n
 Requirements:
 - OpenCV 2.4.X (not 3.0.0 aplha yet)
 - Eigen 3
 - Boost > 1.49
 - Intel TBB

 In addition to the above libraries, this program internally contains code from the following external open-source libraries:
 - A pupil tracker by <a href="http://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/">Lech Swirski</a>
 - LSD (OpenCV 3.0a): a Line Segment Detector by <a href="http://www.ipol.im/pub/art/2012/gjmr-lsd/">Rafael Grompone von Gioi (We use an OpenCV-3.0 version of the BSD license)</a>

 The former is slightly modified (PupilTracker.cpp) and the latter is used as is.

 @section overview Overview
 main.cpp should give you a brief usage.\n
 In short, there are two steps to detect eyeball position from an image:
 1. eye_tracker::IrisDetector::DetectIris detects iris region and outputs a 2D iris ellipse (eye_tracker::Ellipse), then
 2. the ellipse computes the eyeball position given a few eyeball parameters (eye_tracker::Ellipse::ComputeEllipsePose).
 
 eye_tracker::IrisDetector is in iris_detector.h and eye_tracker::Ellipse in iris_geometry.h

 By default abundant debug output are enabled by preprocessor macros (DEBUG_*) in iris_detector.cpp and iris_geometry.h.

 Run main.exe in ./bin (vc10, x86 build) to see an example image result.
 
 @section issues Known Issue
 Only <b>Release</b> build works properly due to a certain error in PupilTracker.cpp steming from TBB.

 @section license License Issue
 Due to conflicting licenses in the external libraries used in this code, I have not made this project open to the public, yet.
 Thus please use this code for now as a reference and do not redistribute it publically. 
 I will resolve this issue and make the entire project open source in the near future.

 @author Yuta Itoh <itoh@in.tum.de>, \n<a href="http://wwwnavab.in.tum.de/Main/YutaItoh">Homepage</a>.

**/


#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Dense>

#include "iris_detector.h"


int main (int argc,char *argv[]){
	


	/// logger from "PupilTracker.h".
	/// to turn off the logging, comment out "#define TRACKER_LOG" in external/Swirski/pupiltracker/tracker_log
	tracker_log log;
	
	/// //////////////////////////////////////////////////////
	/// Step 0: Load an eye image and set camera parameters
	/// //////////////////////////////////////////////////////
	bool is_resize_on = true;
	const int kResizeDivizer = 4; //8

	cv::Mat img;
	cv::Mat K; /// Camera intrinsic matrix
	cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
	const std::string img_file = "./eye_img00054.png";
	img = cv::imread(img_file);//C:/Users/Yuta/Copy/20140422_DatasetForAlex/Data/20140706_SPAAMEyeSamples/Yuta1/eye_img00023.png");
	K = (cv::Mat_<double>(3,3) <<
		2095.400390625, 0, 769.07733154296875,
		 0, 2103.191650390625, 556.17303466796875, 
		 0, 0, 1);
	distCoeffs.at<double>(0,0) =  -0.029856393113732338;
	distCoeffs.at<double>(1,0) =   -0.64231938123703003;
	distCoeffs.at<double>(2,0) = -0.0031928007956594229;
	distCoeffs.at<double>(3,0) =  0.0034734096843749285;
	distCoeffs.at<double>(4,0) =     66.245414733886719;
	distCoeffs.at<double>(5,0) =   -0.20115898549556732;
	distCoeffs.at<double>(6,0) =    0.86070114374160767;
	distCoeffs.at<double>(7,0) =     63.783740997314453;

	if( img.empty() ){
		std::cerr<< "Failed to read image: "<<img_file<<std::endl;
		return -1;
	}
	cv::imshow("Raw input image",img);

	

	cv::Mat mapx, mapy;
	SECTION("main: Image undistortion", log)
	{
		cv::initUndistortRectifyMap(K, distCoeffs, cv::Mat(), K, img.size(), CV_32FC1, mapx, mapy);
		cv::remap(img, img, mapx, mapy, cv::INTER_LINEAR);
	}
	
	/// Convert K from OpenCV to OpenGL, i.e.,
	/// to Right-handed 3D coordinates with top-left image origin
	Eigen::Matrix3d K_GL;
	K.at<double>(1,1) = -K.at<double>(1,1); // fy -> -fy
	for( size_t r(0); r<3; r++)
		for( size_t c(0); c<3; c++)
			K_GL(r,c)=K.at<double>(r,c);
	K_GL(1,1)=-K_GL(1,1);
	/// Resize Image and modify intrinsic mat
	if (is_resize_on){
		cv::resize(img,img,cv::Size(img.cols/kResizeDivizer,img.rows/kResizeDivizer));
		const double K22 = K_GL(2,2);
		K_GL=K_GL/kResizeDivizer;
		K_GL(2,2)=K22;
	}

	
	/// //////////////////////////////////////////////////////
	/// Step 1: Detect iris ellipse from an image
	/// //////////////////////////////////////////////////////
	eye_tracker::IrisDetector iris_detector;
	SECTION("main: DetectIris", log)
	{
		bool eyeball_found = iris_detector.DetectIris(img);
		if ( eyeball_found == false){
			std::cerr << "No eye ball found, skip this image" <<std::endl;
		}
	}
	
	/// //////////////////////////////////////////////////////
	/// Step 2: Compute 3D eye position based on detected 2D ellipse and Limbus radius
	/// //////////////////////////////////////////////////////
	eye_tracker::Ellipse ellipse = iris_detector.get_ellipse();
	std::vector<Eigen::Vector3d> limbus_positions;
	std::vector<Eigen::Vector3d> eye_positions;
	std::vector<Eigen::Vector3d> eye_gazes;
	const double rL = 0.0055;/// Limbus radius [m]
	SECTION("main: ComputeEllipsePose", log)
	{
		bool is_eye_position_computed = ellipse.ComputeEllipsePose(K_GL, rL, img,
			limbus_positions,
			eye_positions,
			eye_gazes
			);

		if( is_eye_position_computed == false ){
			std::cout << "The ellipse does not give a valid eye position." <<std::endl;
			return false;
		}
	}

	log.print();
	
	/// Display estimation result
	std::cout<< std::endl;
	std::cout<< "Result:"<<std::endl;
	std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
	for( size_t i=0; i<eye_positions.size(); i++){
		std::cout<< "Eye position candidates ["<<i<<"]: "<<std::setprecision(15)<<eye_positions[i].transpose()<<std::endl;
	}
	cv::waitKey(-1);

	return 0;
}
