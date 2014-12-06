#ifndef __PUPILTRACKER_H__
#define __PUPILTRACKER_H__

#include <vector>
#include <string>

#include <boost/lexical_cast.hpp>
#include <opencv2/core/core.hpp>

#include "timer.h"
#include "ConicSection.h"
#include "tracker_log.h"


namespace PupilTracker {
	
	struct TrackerParams
	{
		int Radius_Min;
		int Radius_Max;

		double CannyBlur;
		double CannyThreshold1;
		double CannyThreshold2;
		int StarburstPoints;

		int PercentageInliers;
		int InlierIterations;
		bool ImageAwareSupport;
		int EarlyTerminationPercentage;
		bool EarlyRejection;
		int Seed;
	};

	const cv::Point2f UNKNOWN_POSITION = cv::Point2f(-1,-1);

	struct EdgePoint
	{
		cv::Point2f point;
		double edgeStrength;

		EdgePoint(const cv::Point2f& p, double s) : point(p), edgeStrength(s) {}
		EdgePoint(float x, float y, double s) : point(x,y), edgeStrength(s) {}

		bool operator== (const EdgePoint& other)
		{
			return point == other.point;
		}
	};

	struct findPupilEllipse_out {
		cv::Rect roiHaarPupil;
		cv::Mat_<uchar> mHaarPupil;
		
		cv::Mat_<float> histPupil;
		double threshold;
		cv::Mat_<uchar> mPupilThresh;

		cv::Rect bbPupilThresh;
		cv::RotatedRect elPupilThresh;

		cv::Rect roiPupil;
		cv::Mat_<uchar> mPupil;
		cv::Mat_<uchar> mPupilOpened;
		cv::Mat_<uchar> mPupilBlurred;
		cv::Mat_<uchar> mPupilEdges;
		cv::Mat_<float> mPupilSobelX;
		cv::Mat_<float> mPupilSobelY;

		std::vector<EdgePoint> edgePoints;
		std::vector<cv::Point2f> inliers;
		int ransacIterations;
		int earlyRejections;
		bool earlyTermination;

		cv::Point2f pPupil;
		cv::RotatedRect elPupil;

		float A,B,C,D,E,F;

		findPupilEllipse_out() : pPupil(UNKNOWN_POSITION), threshold(-1) {}
	};
	bool findPupilEllipse(
		const TrackerParams& params,
		const cv::Mat& m,

		std::vector<cv::Point2f> &edgePoints, /// added

		findPupilEllipse_out& out,
		tracker_log& log
		);

}//PupilTracker





#endif//__PUPILTRACKER_H__
