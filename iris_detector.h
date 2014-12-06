#ifndef IRIS_DETECTOR_H
#define IRIS_DETECTOR_H

/**
 @author Yuta Itoh <itoh@in.tum.de>, \n<a href="http://wwwnavab.in.tum.de/Main/YutaItoh">Homepage</a>.
**/


#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>
#include "iris_geometry.h"

#include "pupiltracker/tracker_log.h"

namespace eye_tracker{
	

/**
@class IrisDetector
@brief A class for detecting Detects an iris ellipse in a given image
See IrisDetector::DetectIris for more detail.
*/
class IrisDetector
{
public:
	IrisDetector(){
	};
	~IrisDetector(){
		ClearEdgeLists();
	}
	bool DetectIris(cv::Mat &src);
	
	Ellipse get_ellipse(){return ellipse_;};
private:
	IrisDetector(const IrisDetector& other);
	IrisDetector& operator=(const IrisDetector& rhs);
	
	bool DetectIrisRegion(const cv::Mat &src, cv::Mat &src_eye,
		cv::Point2f &iris_pos_in_roi, cv::Point &iris_roi_offset,
		std::vector<cv::Point2f> &edgePoints);
	// Main function
	bool FindLimbus( const cv::Mat &src_img, const cv::Point2f &iris_position_in_roi, const cv::Point &iris_roi_offset,
		std::vector<cv::Point2f> &edgePoints);

	void RegisterImage( const cv::Mat &src_img );
	void MorphologicalPreprocessing();
	void ExtractEdges(); /// Apply Canny edge detector
	void ExtractEdgesByLSD(const cv::Point2f &iris_position_in_roi); /// Appply Line Segment detector by http://www.ipol.im/pub/art/2012/gjmr-lsd/
	double minDitanceBetweenLines( double* a, double *b );

	/// Stuff for edge clustering
	static bool ComparePoint2iVecPtrPredicate( const EdgeList* lhs, const EdgeList* rhs ){
//		const int lhs_len = std::abs<int> ( (*lhs)[lhs->size()-1].y - (*lhs)[0].y );
//		const int rhs_len = std::abs<int> ( (*rhs)[rhs->size()-1].y - (*rhs)[0].y );
//		return ( lhs_len > rhs_len );
		return ( lhs->size() > rhs->size() );
	}
	inline void ClearEdgeLists(){ for( unsigned int i=0; i<edge_lists_.size(); i++ ) delete edge_lists_[i];}

	void RefineEdgeList(const EdgeList &src, EdgeListF &dst);
	
	std::vector< EdgeList* > edge_lists_;

	/// Images
	cv::Mat src_img_;
	cv::Mat src_gray_;
	cv::Mat src_gray_blur_;
	cv::Mat src_sobel_x_;
	cv::Mat src_sobel_y_;

	/// Stuff for Elippse fitting
	Ellipse ellipse_;

	/// Logging
	tracker_log log;
};



} /// namespace eye_tracker
#endif // IRIS_DETECTOR_H