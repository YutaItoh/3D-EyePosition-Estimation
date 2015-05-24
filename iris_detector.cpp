/**
 @author Yuta Itoh <itoh@in.tum.de>, \n<a href="http://wwwnavab.in.tum.de/Main/YutaItoh">Homepage</a>.
**/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv/cvaux.h>

/// ///////////////////////////////////////////
/// Preprocessor define words for debbugging
/// ///////////////////////////////////////////

///#define USE_OLD_LSD /// Use an LSD implementation before OpenCV 3.0
#if 1 /// Turn on debug images
 #define DEBUG_PUPILDETECTION
 #define DEBUG_PREPROCESSING
 #define DEBUG_FIND_IRIS
/// #define TEST_INT_EDGES
 #define REFINE_DEBUG
 #define DEBUG_LSD
#endif

// #define ADD_EDGE_BY_PUPIL_TRACKER

#ifdef USE_OLD_LSD
#include "lsd.h"
#else
#include "lsd_opencv3.h"
#endif // USE_OLD_LSD
#include "iris_detector.h"
#include "iris_geometry.h"


/// Taken from Leszek-pupil-tracker ///
#include "pupiltracker/cvx.h"
#include "pupiltracker/PupilTracker.h"
#include "pupiltracker/tracker_log.h"


namespace eye_tracker{


inline int EdgeLeftOrRight(const EdgeList &points, const int x ){
	if(points.size()==0)return 0;
	if(points[points.size()/2].x<x)return -1;
	if(points[points.size()/2].x>x)return  1;
	return 0;
}


/**
* @brief Detect iris ellipse from a given image
*/
bool IrisDetector::DetectIris(cv::Mat &src)
{

	cv::Mat src_eye;
	cv::Point2f iris_pos_in_roi;
	cv::Point   iris_roi_offset;
	std::vector<cv::Point2f> edgePoints;
	bool is_iris_detected = false;
	SECTION("iris_detector2.DetectIrisRegion", log)
	{
		is_iris_detected = DetectIrisRegion( src, src_eye, iris_pos_in_roi, iris_roi_offset, edgePoints);
	}
	if( is_iris_detected == false ){
		return false;
	}
	

	SECTION("iris_detector2.RegisterImage", log)
	{
		RegisterImage( src_eye );
	}


	// Use Line Segment Detector for iris edge extraction
	SECTION("iris_detector2.ExtractEdgesByLSD", log)
	{
		ExtractEdgesByLSD(iris_pos_in_roi);
	}

	bool is_ellipse_found = false;
	SECTION("iris_detector2.FindLimbus", log)
	{
		is_ellipse_found = FindLimbus( src_eye, iris_pos_in_roi, iris_roi_offset, edgePoints );
	}


	// Print the timer log
	log.print();

	return is_ellipse_found;
}

bool IrisDetector::DetectIrisRegion(const cv::Mat &src, cv::Mat &src_eye, 
	cv::Point2f &iris_pos_in_roi, cv::Point &iris_roi_offset,
	std::vector<cv::Point2f> &edgePoints){
//	EyeCornerDetector eye_corner(src);
	cv::Point2f iris_pos_in_src;
	
	PupilTracker::findPupilEllipse_out out;
	PupilTracker::TrackerParams params;
	params.Radius_Min = 10;//10
	params.Radius_Max = src.cols;//60

	params.CannyBlur = 1.6;
	params.CannyThreshold1 = 30;
	params.CannyThreshold2 = 50;
	params.StarburstPoints = 0;

	params.PercentageInliers = 40;
	params.InlierIterations = 2;
	params.ImageAwareSupport = true;
	params.EarlyTerminationPercentage = 95;
	params.EarlyRejection = true;
	params.Seed = -1;

#if 1 
	cv::Point clip_topleft(0,0);
	cv::Point clip_buttomright(0,0);
#else /// Clip?
	cv::Point clip_topleft(0,120);
	cv::Point clip_buttomright(250,0);///250
#endif
	cv::Mat src_clipped(src,cv::Rect(0+clip_topleft.x,0+clip_topleft.y,
		         src.cols-clip_topleft.x-clip_buttomright.x,
				 src.rows-clip_topleft.y-clip_buttomright.y));

	/// Code taken and modified from http://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/
	bool is_iris_detected = PupilTracker::findPupilEllipse(params, src_clipped, edgePoints, out, log);
	if( is_iris_detected == false ) return is_iris_detected;

	//  ___________________________________
	// |src                                 | 
	// |     |  |                   　　　  |
	// |                            　　　  |
	// |     |  |                   　　　  |
	// |       clip_topleft         　　　  |
	// | _  _|/_|____________            　 |
	// |     |src_clipped    |       　　 　|
	// |     |  |            |       　　 　|
	// |     |   iris_roi_offset            |
	// | _ _ |_ |/________   |    　　      |
	// |     |  |src_eye  |  |    　　      |
	// |     |  |         |  |    　　      |
	// |     |  |    *    |  |         　 　|
	// |     |  |   iris_pos_in_roi in src_eye or iris_pos_in_src in src
	// |     |  |        |   |              |
	// |     |  |________|   |          　　|
	// |     |               | clip_buttomright (NOTE: this is the distance FROM the "buttom-right" corner)
	// |     |______________ |/_ _ _ _ _ _ _|
	// |                     |              |
	// |                             　　　 |
	// |                     |       　　　 |
	// |                             　　　 |
	// |                     |       　　　 | buttom-right corner
	// |____________________________________|/
	//
	/// Raw detected iris center and ROI
	cv::Point2f pupilCenter = out.elPupilThresh.center;
	cv::Rect roiPupil = out.roiPupil;

	/// Make sure a detected ROI and an iris center stay in the positive coordinates 
	if( roiPupil.x<0 ){
		pupilCenter.x += roiPupil.x;
		roiPupil.width+= roiPupil.x;
		roiPupil.x = 0;
		if(roiPupil.width<0 || roiPupil.x <0){
			std::cerr<<"The detected pupil ROI is invalid ! (It is outside of the original image)"<<std::endl;
			return false;
		}
	}
	if( roiPupil.y<0 ){
		pupilCenter.y  += roiPupil.y;
		roiPupil.height+= roiPupil.y;
		roiPupil.y = 0;
		if(roiPupil.height<0 || roiPupil.y <0){
			std::cerr<<"The detected pupil ROI is invalid ! (It is outside of the original image)"<<std::endl;
			return false;
		}
	}

	iris_pos_in_src = cv::Point2f(pupilCenter.x + clip_topleft.x, 
		                          pupilCenter.y + clip_topleft.y );
	cvtColor( src_eye, src_eye, CV_GRAY2BGR );/// !!! for pupiltracker
	iris_roi_offset = cv::Point(  roiPupil.x + clip_topleft.x,
		                          roiPupil.y + clip_topleft.y);
	iris_pos_in_roi = cv::Point2f(pupilCenter.x - roiPupil.x ,
                                  pupilCenter.y - roiPupil.y);
	/// Expand detected iris region
	cv::Rect eye_rect = roiPupil;
	
	/// Check the final ROI, and crop it within the size of the original image if necessary.
	/// iris_roi_offset and iris_pos_in_roi might also be modified

///	eye_rect.width *=1.5;///1.2;///

	if( eye_rect.x + eye_rect.width > src_clipped.cols ){
		eye_rect.width  = src_clipped.cols - eye_rect.x;
	}
	if( eye_rect.y + eye_rect.height > src_clipped.rows ){
		eye_rect.height = src_clipped.rows - eye_rect.y;
	}
	src_eye = src_clipped(eye_rect);

	if( src_eye.empty() ){
		std::cerr<<"Clipped image is empty!"<<std::endl;
		return false;
	}

#ifdef DEBUG_PUPILDETECTION
//	std::cout<<"roiPupil: "<< out.roiPupil << std::endl;
//	std::cout<<"eye_rect: "<< eye_rect << std::endl;
	std::cout<<"clip_topleft: "<< clip_topleft << std::endl;
	std::cout<<"iris_roi_offset: "<< iris_roi_offset << std::endl;
	std::cout<<"iris_pos_in_roi: "<< iris_pos_in_roi<< std::endl;
	cv::Mat src_clipped_clone = src_clipped.clone();
	cv::Mat src_eye_clone     = src_eye.clone();
	cv::rectangle(src_clipped_clone, eye_rect, cv::Scalar(0,0,200), 3, 4);
	cv::circle(src_clipped_clone, cv::Point(
		static_cast<int>( iris_pos_in_src.x-clip_topleft.x ),
		static_cast<int>( iris_pos_in_src.y-clip_topleft.y ) ),
						  2, cv::Scalar(0,0,200), 2, 4);
	cv::circle(src_eye_clone,     cv::Point(iris_pos_in_roi), 4, cv::Scalar(0,200,0), 2, 4);
	cv::imshow("src_clipped",src_clipped_clone);
	cv::imshow("src_eye",src_eye_clone);
#endif // DEBUG_PUPILDETECTION
	return true;
}

void IrisDetector::MorphologicalPreprocessing( ){

	int morph_size = 1;///2
	cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

	/// Apply the specified morphology operation
	cv::morphologyEx( src_img_, src_img_, cv::MORPH_OPEN, element );
}

void IrisDetector::RegisterImage( const cv::Mat &src_img ){
	ClearEdgeLists();
	src_img_ = src_img.clone(); // Clone!!
	
	MorphologicalPreprocessing();

	cv::Mat src_blur;
///	const int kBlurSize = ((int)(src_img_.rows/100)+1)*2 + 1;//5
	const int kBlurSize = ((int)(src_img_.rows*0.001)+1)*2 + 1;//5
#if 0 // blur
	GaussianBlur( src_img_, src_blur, cv::Size(kBlurSize,kBlurSize), 0, 0, cv::BORDER_DEFAULT );
	cvtColor( src_blur, src_gray_, CV_BGR2GRAY );
#else
	cvtColor( src_img_, src_gray_, CV_BGR2GRAY );
	cv::equalizeHist( src_gray_, src_gray_);

#if 0 // Remove highlight
	cv::Scalar meanVal = cv::mean( src_gray_ ); // costly...
	meanVal *=1.5;
	if(meanVal.val[0]>255.0) meanVal.val[0] = 230;
	const uchar kThreshold=(uchar) meanVal.val[0];
//	cv::threshold(mEye,mEye,230,255,cv::THRESH_TOZERO_INV);
	for(int r=0;r<src_gray_.rows;r++){
		uchar fill_color = src_gray_.at<uchar>(r,0);
		if(fill_color>kThreshold) fill_color=kThreshold;
		for(int c=0;c<src_gray_.cols;c++){
			if(src_gray_.at<uchar>(r,c)>kThreshold){
				src_gray_.at<uchar>(r,c)=fill_color;
			}else{
				fill_color=src_gray_.at<uchar>(r,c);
			}
		}
	}
	cv::imshow("Gray",src_gray_);
#endif

#endif

#ifdef DEBUG_PREPROCESSING
	cv::imshow("MORPH_AFTER_BLUR",src_gray_);
#endif // DEBUG_PREPROCESSING

#if 0
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	clahe->apply(src_gray_,src_gray_);
	imshow("lena_CLAHE",src_gray_);
#endif

	///		cv::medianBlur( src_gray_, src_gray_blur_, 3 );
	//		cv::Sobel(src_gray_blur_, src_sobel_x_, CV_32F,1,0);
	//		cv::Sobel(src_gray_blur_, src_sobel_y_, CV_32F,0,1);
	cv::Sobel(src_gray_, src_sobel_x_, CV_32F,1,0);
	cv::Sobel(src_gray_, src_sobel_y_, CV_32F,0,1);

	///		bitwise_xor(src_gray_, cv::Scalar::all(255), src_gray_);
	/// Reduce noise with a kernel 3x3
	///	blur( src_gray_, src_gray_blur_, cv::Size(3,3) );
	///		cv::imshow( "Blured", src_gray_blur_ );
}


/*
int subpixSampleFast( const Image& src, const Math::Vector< 2, float >& p )
{
	int x = static_cast< int >( floorf( p( 0 ) ) );
	int y = static_cast< int >( floorf( p( 1 ) ) );
	int dx = static_cast< int >( 256 * ( p( 0 ) - floorf( p( 0 ) ) ) );
	int dy = static_cast< int >( 256 * ( p( 1 ) - floorf( p( 1 ) ) ) );
	unsigned char* i = reinterpret_cast< unsigned char* >( src.imageData ) + y * src.colsStep + x;
	int a = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
	i += src.colsStep;
	int b = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8) ;
	return a + ( ( dy * ( b - a ) ) >> 8 );
}
*/
/*
int subpixSampleSafe ( const IplImage* pSrc, CvPoint2D32f p )
{
	int x = int( floorf ( p.x ) );
	int y = int( floorf ( p.y ) );

	if ( x < 0 || x >= pSrc->width  - 1 ||
		 y < 0 || y >= pSrc->height - 1 )
		return 127;

	int dx = int ( 256 * ( p.x - floorf ( p.x ) ) );
	int dy = int ( 256 * ( p.y - floorf ( p.y ) ) );

	unsigned char* i = ( unsigned char* ) ( ( pSrc->imageData + y * pSrc->widthStep ) + x );
	int a = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
	i += pSrc->widthStep;
	int b = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
	return a + ( ( dy * ( b - a) ) >> 8 );
}
*/
int subpixSampleSafe( const cv::Mat& src, const float px, const float py )
{
	int x = static_cast< int >( floorf( px ) );
	int y = static_cast< int >( floorf( py ) );

	if ( x < 0 || x >= src.cols - 1 || y < 0 || y >= src.rows - 1 )
	{
		// continue border pixels
		if ( x < 0 ) 
			x = 0;
		else if ( x >= src.cols )
			x = src.cols - 1;
			
		if ( y < 0 )
			y = 0;
		else if ( y >= src.rows )
			y = src.rows - 1;
			
		return *( reinterpret_cast< unsigned char* >( src.data ) + y * src.step + x );
	}
	
	// do normal sampling
	int dx = static_cast< int >( 256 * ( px - floorf( px ) ) );
	int dy = static_cast< int >( 256 * ( py - floorf( py ) ) );
	unsigned char* i = reinterpret_cast< unsigned char* >( src.data ) + y * src.step + x;
	int a = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
	i += src.step;
	int b = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
	return a + ( ( dy * ( b - a ) ) >> 8 );
}
float subpixSampleSafeF( const cv::Mat& src, const float px, const float py )
{
	int x = static_cast< int >( floorf( px ) );
	int y = static_cast< int >( floorf( py ) );

	if ( x < 0 || x >= src.cols - 1 || y < 0 || y >= src.rows - 1 )
	{
		// continue border pixels
		if ( x < 0 ) 
			x = 0;
		else if ( x >= src.cols )
			x = src.cols - 1;
			
		if ( y < 0 )
			y = 0;
		else if ( y >= src.rows )
			y = src.rows - 1;
			
		return *( src.ptr<float>(y) + x );
	}
	
	// do normal sampling
	float dx = static_cast< float >( ( px - floorf( px ) ) );
	float dy = static_cast< float >(  ( py - floorf( py ) ) );
	const float* i = src.ptr<float>(y) + x;
	float a = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) )  );
	i = src.ptr<float>(y+1) + x;
	float b = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) )  );
	return a + ( ( dy * ( b - a ) )   );
}


/**
* @function RefineEdgeList
* @brief 
*/
void IrisDetector::RefineEdgeList(const EdgeList &src, EdgeListF &dst){

#ifdef REFINE_DEBUG
	const size_t kN=src.size();
	const float kImageScale=2;
///	cv::Mat img = src_sobel_x_.clone();
	static cv::Mat img;
	if( img.empty() ||
		(img.rows != src_gray_.rows*kImageScale || img.cols != src_gray_.cols*kImageScale)
		)
	{
		img = src_gray_.clone();
		cv::cvtColor(img,img,CV_GRAY2RGB);
		cv::resize(img,img,img.size()*(int)kImageScale);
	}
	for( size_t i=0; i<src.size(); i++){
		cv::circle(img, src[i]*kImageScale, 1*kImageScale, cv::Scalar(0,0,200), 1, 4);
	}
#endif
	const double img_center_x = src_img_.cols/2.0;
	const double img_center_y = src_img_.rows/2.0;

	int valid_id = 0;
	dst.resize(src.size());
	const float kSearchPixelRange=src_img_.rows/120.0f;
	const double kThreshold_rad = 25.0/180*M_PI;
	for( size_t i=0; i<src.size(); i++){
		const int x=src[i].x;  
		const int y=src[i].y;
		const float dx=src_sobel_x_.at<float>(y,x);
		const float dy=src_sobel_y_.at<float>(y,x);

		/// check if a gradient vector is not heading to the image center
		const double tmp_x = x-img_center_x;
		const double tmp_y = y-img_center_y;
		const double inner_prod_val = tmp_x*dx + tmp_y*dy;
		const double d_norm   = sqrt(dx*dx+dy*dy);
		const double tmp_norm = sqrt(tmp_x*tmp_x+tmp_y*tmp_y);
		const double angle_rad = acos(inner_prod_val/(d_norm*tmp_norm));
		if( abs(angle_rad)>kThreshold_rad) continue;

		/// remove a point with a high y-axis gradient since it can be an eye lid region
		if( abs(dx)<1e-10 || abs(dy/dx)>0.8 ) continue; 
//		if( abs(dx)<1e-10 || abs(dy/dx)>1.3 ) continue; 

		const float dn=1.0f/d_norm*kSearchPixelRange;
		const int kSampleNum=3;
		Eigen::MatrixXf X(3,2*kSampleNum+1);
		Eigen::MatrixXf Y(2*kSampleNum+1,1);
#ifdef REFINE_DEBUG
		cv::Point ps(kImageScale*((float)x + dx*(-kSampleNum)*dn),kImageScale*((float)y + dy*(-kSampleNum)*dn));
		cv::Point pe(kImageScale*((float)x + dx*( kSampleNum)*dn),kImageScale*((float)y + dy*( kSampleNum)*dn));
		cv::line(img, ps, pe, cv::Scalar(0,0,200), 1, 4);
		cv::circle(img, ps, 1*kImageScale, cv::Scalar(0,200,0), 1, 4);
		cv::circle(img, pe, 1*kImageScale, cv::Scalar(200,0,200), 1, 4);
		double val_sum = 0.0;
#endif
		/// 
		for(int k=-kSampleNum;k<=kSampleNum;k++){
			const float xf=(float)x + dx*k*dn;
			const float yf=(float)y + dy*k*dn;
			const float valx=subpixSampleSafeF(src_sobel_x_,xf,yf);
			const float valy=subpixSampleSafeF(src_sobel_y_,xf,yf);
			X(0,k+kSampleNum)=static_cast <float> (k*k);
			X(1,k+kSampleNum)=static_cast <float> (k);
			X(2,k+kSampleNum)=1.0f;
			Y(k+kSampleNum,0)=sqrt(valx*valx+valy*valy);
///			Y(k+kSampleNum,0)=valx;
#ifdef REFINE_DEBUG
			val_sum += Y(k+kSampleNum,0);
			const int xx = kImageScale*xf;
			const int yy = kImageScale*yf;

			if(xx>=0&&xx<img.cols&&yy>=0&&yy<img.rows){
				std::cout<< k <<" "<<valx<<std::endl;
			}
#endif
		}
		

		Eigen::Matrix3f X0 = X*X.transpose();
		Eigen::MatrixXf A = X0.inverse()*X*Y;
#ifdef REFINE_DEBUG
		val_sum/=(2*kSampleNum+1);
///		if(val_sum>20.0) continue;
		std::cout<< "Val sum " <<val_sum<<std::endl;
		std::cout<< "A " <<A<<std::endl;
#endif
///		A = X0.inverse()*A;
		const float peak_dn=-0.5f*A(1,0)/A(0,0);
		float x_new = (float)x + dx*peak_dn*dn;
		float y_new = (float)y + dy*peak_dn*dn;
		
#ifdef REFINE_DEBUG
		std::cout<< "egde refine " << x<<" "<< x_new<<",  "<< y<<" "<<y_new<<std::endl;
#endif
		if(x_new!=x_new) x_new=static_cast <float>(x);
		if(y_new!=y_new) y_new=static_cast <float>(y);
#ifdef REFINE_DEBUG
		cv::circle(img, cv::Point(x_new*kImageScale,y_new*kImageScale),
			1*kImageScale, cv::Scalar(0,200,0), 1, 4);
		//cv::imshow("RefineEsgeList",img);
		//cv::waitKey(-1);
#endif
		if(1){
			dst[valid_id].x = x_new;
			dst[valid_id].y = y_new;
			valid_id++;
		}
		// subpixel localisation of maximum: fit 3 pixels to parabola and compute zero of derivation
		
	}
	dst.resize(valid_id);
	
#ifdef REFINE_DEBUG
		cv::imshow("RefineEsgeList",img);
		cv::waitKey(-1);
#endif
}


/**
* @function FindLimbus
* @brief The main function for iris detection
*/
bool IrisDetector::FindLimbus( const cv::Mat &src_img, const cv::Point2f &iris_position_in_roi, const cv::Point &iris_roi_offset,
	std::vector<cv::Point2f> &edgePoints
	){
	
	const double kImageSize= src_img.rows*src_img.cols;
	static bool is_learning_on = false;
	
	cv::Mat src_with_ellipses;
	src_img_.copyTo( src_with_ellipses );
#ifdef TEST_INT_EDGES
	cv::Mat src_with_ellipses2;
	src_img_.copyTo( src_with_ellipses2 );
#endif // TEST_INT_EDGES


	if( edge_lists_.size() < 2 ){
		return false;
	}

	cv::Mat img_with_best_ellipse;
	src_img_.copyTo( img_with_best_ellipse);

	const int kCandNumMax = 10; /// Top kCandNumMax edge segments will be used

	int cand_num = edge_lists_.size();
	if( cand_num > kCandNumMax ) cand_num=kCandNumMax;
	double best_similarity = DBL_MAX;

	std::vector< EdgeListF > edge_lists_float(edge_lists_.size());
	SECTION("FindLimbus.RefineEdges", log)
	{
		for( size_t k=0; k<edge_lists_.size(); k++){
			RefineEdgeList( *edge_lists_[k], edge_lists_float[k]);
		}
	} /// SECTION

	SECTION("FindLimbus.SearchEllipse", log)
	{
#ifdef ADD_EDGE_BY_PUPIL_TRACKER /// Add edges found by PupilTracker.cpp
		edge_lists_float.push_back(edgePoints); 
		EdgeList* edgePointsInt = new EdgeList;  /// This int version will never used and deleted at ClearEdgeLists...
		edgePointsInt->resize(edgePoints.size());/// thus, just for debugging...
		for( int i=0; i<edgePointsInt->size(); i++){
			(*edgePointsInt)[i].x = (int)edgePoints[i].x;
			(*edgePointsInt)[i].y = (int)edgePoints[i].y;
		}
		edge_lists_.push_back(edgePointsInt); /// Edges found by PupilTracker.cpp
		cand_num = edge_lists_float.size();;
#endif /// Add edges found by PupilTracker.cpp

#ifdef DEBUG_RANSAC
		ellipse_.SetDebugMat(src_img_);
#endif // RANSAC_DEBUG
#ifdef TEST_INT_EDGES
		Ellipse ellipse2;
#endif // TEST_INT_EDGES
#ifdef DEBUG_FIND_IRIS
		///		cv::imwrite("cand0.png",img_with_best_ellipse);
#endif //  DEBUG_FIND_IRIS

//#define USE_EDGE_PAIRS
#ifdef USE_EDGE_PAIRS // Edge pairs
		int best_i = -2;
		int best_j = -1;
		int best_inlier_num = 0;
		const int kHaarIrisCenterX = static_cast <int> (iris_position_in_roi.x);
		const int kHaarIrisCenterY = static_cast <int> (iris_position_in_roi.y);
		const int kHaarIrisCenterThreshold = (src_img_.rows/5)*(src_img_.rows/5);
		for( int i=-1; i<cand_num; i++){
			for( int j=i+1; j<cand_num; j++){

				int ellipse_fitting_inlliers;
				if( i== -1) {
					ellipse_fitting_inlliers = ellipse_.FitEllipse( edge_lists_float[j], src_sobel_x_, src_sobel_y_ );
#ifdef TEST_INT_EDGES
					ellipse2.FitEllipse( *edge_lists_[j], src_sobel_x_, src_sobel_y_ );
#endif // TEST_INT_EDGES
				}else{
					/*
					const int label1 = EdgeLeftOrRight(*edge_lists_[i], iris_position_in_roi.x);
					const int label2 = EdgeLeftOrRight(*edge_lists_[j], iris_position_in_roi.x);
					if(label1==-1&&label2==-1)continue;
					if(label1== 1&&label2== 1)continue;
					*/

					///		SECTION("FindLimbus.SearchEllipse.EllipseFit", log)
					{
						ellipse_fitting_inlliers = ellipse_.FitEllipse( edge_lists_float[i], edge_lists_float[j], src_sobel_x_, src_sobel_y_  );
					}
#ifdef TEST_INT_EDGES
					ellipse2.FitEllipse( *edge_lists_[i], *edge_lists_[j], src_sobel_x_, src_sobel_y_ );
#endif // TEST_INT_EDGES
				}
				if(ellipse_fitting_inlliers == 0) continue;

#ifdef TEST_INT_EDGES
				ellipse2.DrawEllipse(src_with_ellipses2);
#endif // TEST_INT_EDGES

#ifdef DEBUG_FIND_IRIS
				ellipse_.DrawEllipseManual(src_with_ellipses,0.0);
#endif //  DEBUG_FIND_IRIS
				/// Remove outliers as much as possible, we use many heuristics, though
				const int kIrisCenterDiff =  static_cast<int>(
					(ellipse_.x()-kHaarIrisCenterX)*(ellipse_.x()-kHaarIrisCenterX) 
					+(ellipse_.y()-kHaarIrisCenterY)*(ellipse_.y()-kHaarIrisCenterY)
					);
				///if( kIrisCenterDiff>kHaarIrisCenterThreshold ) continue;
				///				if( ellipse_.r() < 0.59 ) continue;// 0.79
				///				if( ellipse_.BoundingArea() >kImageSize*0.9 )continue;// 0.2
				if( ellipse_.BoundingArea() <kImageSize*0.1 )continue;// 0.1
				///std::cout<<ellipse_.BoundingArea()<<std::endl;
				///				if( ellipse_.CenterSquaredDistance(iris_position_in_roi) >350 ) continue;
				///ellipse_.DrawEllipse(img_with_best_ellipse);
#ifdef DEBUG_FIND_IRIS
				ellipse_.DrawEllipseManual(src_with_ellipses,1.0);
#endif //  DEBUG_FIND_IRIS


				if( best_inlier_num< ellipse_fitting_inlliers ) {
					best_inlier_num = ellipse_fitting_inlliers;
					best_i=i;
					best_j=j;
				}

			}
		}


		if(best_i==-2){
			return false;
		}

		if( best_i== -1) {
			ellipse_.FitEllipse( edge_lists_float[best_j], src_sobel_x_, src_sobel_y_ );
		}else{
			ellipse_.FitEllipse( edge_lists_float[best_i], edge_lists_float[best_j], src_sobel_x_, src_sobel_y_ );
		}
#endif // USE_EDGE_PAIRS
		/// Use all edge points
		size_t egde_list_total = 0;
		for( size_t i=0; i<edge_lists_float.size(); i++){
			egde_list_total+=edge_lists_float[i].size();
		}
		size_t k=0;
		EdgeListF edge_lists_float_all(egde_list_total);
		for( size_t i=0; i<edge_lists_float.size(); i++){
			for( size_t j=0; j<edge_lists_float[i].size(); j++){
				edge_lists_float_all[k].x=edge_lists_float[i][j].x;
				edge_lists_float_all[k].y=edge_lists_float[i][j].y;
				k++;
			}
		}
		cv::Vec3b tmp;
		tmp[1]=255;
		for( size_t i=0; i<edge_lists_float_all.size(); i++){
			const cv::Point2f &vec=edge_lists_float_all[i];
			const int x = vec.x;
			const int y = vec.y;
			if( 0<=x && x<src_with_ellipses.cols && 0<=y && y<src_with_ellipses.rows ){
				src_with_ellipses.at<cv::Vec3b>(y,x)=tmp;
			}
		}
		std::cout<<"Start: ellipse_.FitEllipse( edge_lists_float_all, src_sobel_x_, src_sobel_y_ );"<<std::endl;
		ellipse_.FitEllipse( edge_lists_float_all, src_sobel_x_, src_sobel_y_ );
		std::cout<<"End: ellipse_.FitEllipse( edge_lists_float_all, src_sobel_x_, src_sobel_y_ );"<<std::endl;
		ellipse_.DrawEllipseManual(src_with_ellipses,0.77);
		

#ifdef DEBUG_FIND_IRIS
		imshow("EllipsesFloat",src_with_ellipses);
#endif // DEBUG_FIND_IRIS
#ifdef  TEST_INT_EDGES
		imshow("EllipsesInt",src_with_ellipses2);
#endif // TEST_INT_EDGES

#if 0
		const double kB=ellipse_.a(); /// 'a' is intended
		ellipse_.SetExplicit(
			ellipse_.x(),
			ellipse_.y(),
			ellipse_.t(),
			ellipse_.a(),
			kB
			);
#endif

		ellipse_.DrawEllipse(img_with_best_ellipse);
		ellipse_.Translate(iris_roi_offset.x,iris_roi_offset.y);
#ifdef DEBUG_FIND_IRIS

		std::cout << "Found iris ellipse"<<std::endl;
		ellipse_.PrintPolynomial();
		ellipse_.PrintExplicit();

		/// Draw edges used for the best ellipse
		cv::Vec3b color;
		for( size_t i=0; i<edge_lists_.size(); i++){
			color = calcPseudoColor( edge_lists_[i]->size()/(double)edge_lists_[0]->size() );
			//		cv::randu(color, cv::Scalar(0), cv::Scalar(255));
			EdgeList *edge_list = edge_lists_[i];
			for( size_t j=0; j<edge_list->size(); j++){
				img_with_best_ellipse.at<cv::Vec3b>( (*edge_list)[j] ) = color;
			}
		}

		std::stringstream ss;
		static int frame=0;

#if 0 /// print params in image
		ss  << "r_: "<<ellipse_.BoundingArea()/kImageSize;
		cv::putText(img_with_best_ellipse, 
			ss.str(), cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,200,0), 1, CV_AA);		
		ss.str("");
		ss.clear();
		ss<<  "a_: " << ellipse_.a();
		cv::putText(img_with_best_ellipse, 
			ss.str(), cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,200,0), 1, CV_AA);		
		ss.str("");
		ss.clear();
		ss<<  "b_: " << ellipse_.b();
		cv::putText(img_with_best_ellipse, 
			ss.str(), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,200,0), 1, CV_AA);		
		ss.str("");
		ss.clear();
		ss<<  "t_: " << ellipse_.t();
		cv::putText(img_with_best_ellipse, 
			ss.str(), cv::Point(10,80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,200,0), 1, CV_AA);					
		ss.str("");
		ss.clear();
		ss<< "r_: " << ellipse_.r();
		cv::putText(img_with_best_ellipse, 
			ss.str(), cv::Point(10,100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,200,0), 1, CV_AA);		
		ss.str("");
		ss.clear();
		ss<<  "frame #: " << frame;
		cv::putText(img_with_best_ellipse, 
			ss.str(), cv::Point(10,120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,200,0), 1, CV_AA);		
#endif

#ifdef USE_EDGE_PAIRS
//		std::ofstream xy_ofs("detected_iris_uv.txt");
		if( best_i== -1) {
			// Do nothing
		}else{
			EdgeListF &e1= edge_lists_float[best_i];
			EdgeListF &e2= edge_lists_float[best_j];
			EdgeList  &e12 = *edge_lists_[best_i];
			EdgeList  &e22 = *edge_lists_[best_j];
			const int kH = img_with_best_ellipse.rows;
			const int kW = img_with_best_ellipse.cols;
			for( size_t k=0; k<e1.size(); k++){
				if( e1[k].x <0 || e1[k].x >kW || e1[k].y <0 || e1[k].y >kH ||
					e12[k].x<0 || e12[k].x>kW || e12[k].y<0 || e12[k].y>kH ) continue;
				img_with_best_ellipse.at<cv::Vec3b>( (int)e1[k].y, (int)e1[k].x ) = cv::Vec3b(0,0,255);
				img_with_best_ellipse.at<cv::Vec3b>(  e12[k].y, e12[k].x ) = cv::Vec3b(0,255,0);
//				xy_ofs <<  e12[k].x << " " <<  e12[k].y <<std::endl;
			}
			for( size_t k=0; k<e2.size(); k++){
				if( e2[k].x <0 || e2[k].x >kW || e2[k].y <0 || e2[k].y >kH ||
					e22[k].x<0 || e22[k].x>kW || e22[k].y<0 || e22[k].y>kH ) continue;
				img_with_best_ellipse.at<cv::Vec3b>( (int)e2[k].y, (int)e2[k].x ) = cv::Vec3b(0,0,255);
				img_with_best_ellipse.at<cv::Vec3b>(  e22[k].y, e22[k].x ) = cv::Vec3b(0,255,0);
//				xy_ofs <<  e22[k].x << " " <<  e22[k].y <<std::endl;
			}
		}
		//		cv::imwrite("cand1.png",img_with_best_ellipse);
#endif // USE_EDGE_PAIRS
		frame++;
#endif // DEBUG_FIND_IRIS

		cv::namedWindow("Candidates" );
		cv::imshow("Candidates", img_with_best_ellipse);


		//		cv::waitKey(-1);

	} /// SECTION

	/// Print log
	log.print();

	return true;
}



//
// cv::Mat img;
//int n_lines;
//double* lines;　
//void change_th_lsd(int nfa, void* dummy)
//{
//    cv::Mat result = img.clone();
//    for(int i = 0; i < n_lines; i++)
//    {
//        const double *line = &lines[i * 7];
//        if(nfa < line[6])
//        {
//            const cv::Point p1(line[0], line[1]);
//            const cv::Point p2(line[2], line[3]);
//            cv::line(result, p1, p2, cv::Scalar(0, 0, 255));
//        }
//    }
//    cv::imshow("result_image", result);
//}
// 


double IrisDetector::minDitanceBetweenLines( double* a, double *b ){
	double ss = abs( a[0] - b[0] ) + abs( a[1] - b[1] );
	double sd = abs( a[0] - b[2] ) + abs( a[1] - b[3] );
	double ds = abs( a[2] - b[0] ) + abs( a[3] - b[1] );
	double dd = abs( a[2] - b[2] ) + abs( a[3] - b[3] );
	return std::min<double>( ss,std::min<double>( ss,std::min<double>(ds,dd) ) );
}

void IrisDetector::ExtractEdgesByLSD(const cv::Point2f &iris_position_in_roi0){
	edge_lists_.clear();
	
	
  /* LSD parameters */
  double scale = 0.6;       /* Scale the image by Gaussian filter to 'scale'. */
  scale = 1.5;//1.2;
//  scale = 2.5;//1.2;
  double sigma_scale = 0.6; /* Sigma for Gaussian filter is computed as
                                sigma = sigma_scale/scale.                    */
  sigma_scale = 1.3	;//1.7;//1.2; /// higher this value, more sensitively detetects edge segments
  double quant = 2.0;       /* Bound to the quantization error on the
                                gradient norm.                                */
  double ang_th = 22.5;     /* Gradient angle tolerance in degrees.           */
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  double density_th = 0.7;  /* Minimal density of region points in rectangle. */
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  
	cv::LineSegmentDetectorImpl lsd_imp(cv::LSD_REFINE_STD, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins);

	const double kAngleThreshold = M_PI/4.0;//9/// ignore edges their angle is less than this
	const int kEdgeSamplingNum = 10;
	Eigen::Vector2d iris_position_in_roi(iris_position_in_roi0.x,iris_position_in_roi0.y);
	cv::Mat img = src_gray_.clone();
	int n_lines;
	double* lines;
#ifdef USE_OLD_LSD
	/// Convert the source image to LSD data structure
	double *dat = new double[img.rows * img.cols];
	SECTION("FindLimbus.ExtractEdgesByLSD.copyImg",log){
		for(int y = 0; y < img.rows; y++)
			for(int x = 0; x < img.cols; x++)
				dat[y * img.cols + x] = img.at<unsigned char>(y, x);
	}
#else
	std::vector<cv::Vec4d> linesVec;
#endif // USE_OLD_LSD

	SECTION("FindLimbus.ExtractEdgesByLSD.LSD",log){
		/// Apply LSD
#ifdef USE_OLD_LSD
		lines = lsd(&n_lines, dat, img.cols, img.rows);


		lines = LineSegmentDetection( &n_lines, dat, img.cols, img.rows, scale, sigma_scale, quant,
			ang_th, log_eps, density_th, n_bins,
			NULL, NULL, NULL );
#else
		std::vector<double> lineWidths;
		std::vector<double> linePrec;
		std::vector<int>    lineNFA;
		lsd_imp.detect(img,linesVec,lineWidths,linePrec,lineNFA);
#endif // USE_OLD_LSD
	}
#ifdef USE_OLD_LSD
	delete dat;
#endif // USE_OLD_LSD
	//// Find maximum and minimum threshold
	//int max_NFA = 0;
	//for(int i = 0; i < n_lines; i++)
	//	max_NFA = std::max(max_NFA, static_cast<int>(lines[i * 7 + 6]));
	//max_NFA = 0;

	/// Variables for edge clustering
	std::vector<double*> sampled_lines;
	std::vector<int> sampled_lines_class;
	const int kUndeterminedClass = -1;

	/// Filter some edges based-on prior knowledge about the eye
	const int nfa = 0;
	SECTION("FindLimbus.ExtractEdgesByLSD.filter",log){
		
#ifndef USE_OLD_LSD
		n_lines=linesVec.size();
#endif

#ifdef DEBUG_LSD
		cv::Mat img_draw = img.clone();
		cv::cvtColor(img_draw, img_draw, CV_GRAY2RGB);
		cv::circle(img_draw, cv::Point(iris_position_in_roi[0],iris_position_in_roi[1]), 2,cv::Scalar(0,255,0),2);
#endif // DEBUG_LSD
		for(int i = 0; i < n_lines; i++)
		{

#ifdef USE_OLD_LSD
			double *line = &lines[i * 7];
#else
			double *line = linesVec[i].val;
#endif // USE_OLD_LSD
			if(true )//||nfa < line[6])
			{
				/// ///////////////////////////
				/// Analyze an edge segment ///
				/// ///////////////////////////

				/// Cut image patch around the segment so that the patch represents:
				///  _____
				/// |     | 
				/// |  A  | <- kHalfPatchWidth
				/// |     | 
				/// o-----o <- the edge
				/// |     |
				/// |  B  |
				/// |     |
				///  -----
				///
				///   * <--- iris_position_in_roi
				///
				/// region B is closer to the estimate iris center than region A

				/// Crop edge-segment region
				cv::RotatedRect rect;
				/// Make an edge vector
				Eigen::Vector2d edge_src;
				Eigen::Vector2d edge_dst;
				/// Convert the edge vector so that the source has lower x than the destination 
				if( line[0]>line[2] ){
					std::swap(line[0],line[2]);
					std::swap(line[1],line[3]);
				}
				edge_src[0] = line[0];
				edge_src[1] = line[1];
				edge_dst[0] = line[2];
				edge_dst[1] = line[3];
#ifdef DEBUG_LSD
				const cv::Point p1(edge_src[0], edge_src[1]);
				const cv::Point p2(edge_dst[0], edge_dst[1]);
				cv::line(img_draw, p1, p2, cv::Scalar(0, 0, 255));
				cv::circle(img_draw, p1, 2,cv::Scalar(0,255,0),1);
				cv::circle(img_draw, p2, 2,cv::Scalar(0,0,255),1);
#endif // DEBUG_LSD

				/// Edge vector
				Eigen::Vector2d edge_vec     = edge_dst - edge_src;
				/// Direction from the edge source to the iris center
				Eigen::Vector2d vec2iris = iris_position_in_roi - edge_src;
					
				const double len = edge_vec.norm();
				double angle     = acos(edge_vec[0]/len);
				if(edge_vec[1]<0.0){
					angle = -angle;
				}
				
				const int kHalfPatchWidth = 25;
				const int kPatchWidth = kHalfPatchWidth*2;;
				cv::Size rect_size(static_cast <int> (len), kPatchWidth);
				const float src_cx = static_cast <float>( (edge_src[0]+edge_dst[0])/2.0 );
				const float src_cy = static_cast <float>( (edge_src[1]+edge_dst[1])/2.0 );
				const float cx = rect_size.width/2.0f -0.5f;
				const float cy = rect_size.height/2.0f-0.5f;


				const double iris_do_prod = -vec2iris[0]*sin(angle)+vec2iris[1]*cos(angle);
				if ( iris_do_prod < 0.0)
				{
					angle += M_PI;
				}
				
				const float cosa = static_cast <float>(cos(angle));
				const float sina = static_cast <float>(sin(angle));
				cv::Matx23f M(  cosa, -sina, -cx*cosa+cy*sina + src_cx,
								sina,  cosa, -cx*sina-cy*cosa + src_cy );
				
				/// 
				cv::Mat img_patch;
				cv::warpAffine(img, img_patch, M, rect_size, 
					cv::INTER_CUBIC | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE );

				/// Analyze the image patch
				/// Region supposed to be outside of the iris
				cv::Mat img_outer_iris = img_patch.rowRange(cv::Range(0,kHalfPatchWidth)) ;
				/// Region supposed to be inside of the iris
				cv::Mat img_inner_iris = img_patch.rowRange(cv::Range(1+kHalfPatchWidth,kPatchWidth)) ;
				const cv::Scalar outer_iris_sum = cv::sum(img_outer_iris);
				const cv::Scalar inner_iris_sum = cv::sum(img_inner_iris);
				/// the inner region should be darker than the outer
				if( ( inner_iris_sum(0) < outer_iris_sum(0) ) &&
					( abs(angle)>kAngleThreshold )
					)
				{
					/// Accept the edge
					sampled_lines.push_back(line);
					sampled_lines_class.push_back(kUndeterminedClass);
#ifdef DEBUG_LSD
					cv::Point pc(  src_cx, src_cy );
					cv::circle(img_draw, pc,4,cv::Scalar(0,255,0),3);
#endif // DEBUG_LSD

				}
				
#ifdef DEBUG_LSD
				else{
					cv::Point pc(  src_cx, src_cy );
					cv::circle(img_draw, pc,4,cv::Scalar(0,0,255),3);
				}
#endif // DEBUG_LSD
#ifdef DEBUG_LSD
				cv::Point p12( -rect_size.height/2*sin(angle)+src_cx, 
								rect_size.height/2*cos(angle)+src_cy );
				cv::Point p13(  rect_size.width/2*cos(angle)+src_cx, 
								rect_size.width/2*sin(angle)+src_cy );
				cv::Point pc(  src_cx, 
							   src_cy );
				cv::Point p22(  rect_size.height/2*sin(angle)+src_cx, 
							   -rect_size.height/2*cos(angle)+src_cy );
				
				cv::line(img_draw, pc, p12, cv::Scalar(255, 0, 255));
				cv::line(img_draw, pc, p22, cv::Scalar(  0, 255, 255));
				cv::line(img_draw, pc, p13, cv::Scalar(  0, 255, 0));/*
				cv::imshow("EdgeSegment", img_patch);
				cv::imshow("result_image", img_draw);*/
				std::cout<<"iris_do_prod  "<<iris_do_prod<<std::endl;
				std::cout<<"edge_vec  "<<edge_vec<<std::endl;
				std::cout<<"a  "<<angle<<std::endl;
				std::cout<<"cx "<<cx<<std::endl;
				std::cout<<"cy "<<cy<<std::endl;
				std::cout<<"src_cx "<<src_cx<<std::endl;
				std::cout<<"src_cy "<<src_cy<<std::endl;
///				std::cout<<"M "<<M<<std::endl;
///				std::cout<<"Inner sum: "<<inner_iris_sum<<std::endl;
///				std::cout<<"Outer sum: "<<outer_iris_sum<<std::endl;
///				cv::waitKey(-1);
#endif // DEBUG_LSD

			}
		}
#ifdef DEBUG_LSD
		cv::imshow("result_image", img_draw);
#endif // DEBUG_LSD
	} /// SECTION


	SECTION("FindLimbus.ExtractEdgesByLSD.edge_clustering",log){
		/// Cluster the line segments by their distances
		const size_t kLineSegmentNum = sampled_lines.size();
		std::vector< std::vector<int> > classes(kLineSegmentNum);
		const double kLinesMinDistThd = (img.rows+img.cols)*0.05; /// [pixel]
		int class_num_count = 0;
		/// Initialize variables
		for ( size_t k=0; k<kLineSegmentNum; k++ ){
			int &class1 = sampled_lines_class[k];
			class1 = k;
			classes[k].push_back(k);
		}
		/// Do clustering
		for ( size_t k=0; k<kLineSegmentNum; k++ ){
			const int kClass1 = sampled_lines_class[k];
			/// Check the distance of k-th line segment and 
			/// the rest of segments behind the k-th
			/// to decide if they should be merged into the same class.
			for ( size_t j=k+1; j<kLineSegmentNum; j++ ){
				const int kClass2 = sampled_lines_class[j];
				if( kClass1 == kClass2) continue;  /// j-th segment belongs to the same class, skip it
				/// compute the minimum distance between "end points" of the two edges
				const double min_dis = minDitanceBetweenLines( sampled_lines[k], sampled_lines[j] );
				/// If two lines are close enough, merger their classes
				if( min_dis < kLinesMinDistThd ){
					/// move line segment indice in the class 1 to class 2
					classes[kClass1].insert(classes[kClass1].end(),
						classes[kClass2].begin(),classes[kClass2].end());
					/// update class references of each segments that were in class 2
					for( size_t m=0;m<classes[kClass2].size();m++){
						sampled_lines_class[ classes[kClass2][m] ] = kClass1;
					}
					/// delete class 2, after all unused classes[x] becomes an empty set
					classes[kClass2].clear();
				}
			}
		}

		/// Pass edge clusters to the main data structure
#ifdef DEBUG_LSD
		cv::Mat img_draw = img.clone();
		cv::cvtColor(img_draw, img_draw, CV_GRAY2RGB);
		cv::Mat img_draw2 = img.clone();
		cv::cvtColor(img_draw2, img_draw2, CV_GRAY2RGB);
		for ( size_t k=0; k<kLineSegmentNum; k++ ){
			const double *line = sampled_lines[k];
			cv::line(img_draw, cv::Point(line[0],line[1]), cv::Point(line[2],line[3]),
				cv::Scalar(0, 255,0),3);
		}
#endif // DEBUG_LSD
		for ( size_t k=0; k<kLineSegmentNum; k++ ){
			const std::vector<int> &line_indice = classes[k];
			const size_t kClusterSize = line_indice.size();
			if( kClusterSize == 0 ) continue;
			EdgeList *edge_points = new EdgeList;
			/// For each line segments in a class,
			/// sample 2D points on the line and store it
			for ( size_t j=0; j<kClusterSize; j++ ){
				const int line_idx = line_indice[j];
				const double *line = sampled_lines[line_idx];
				/// Sample points on the edge segment
				/// first, compute direction vector
				/// from the source point (line[0],line[1]) 
				/// to the destination (line[2],line[3])
				double dx = line[0]-line[2];
				double dy = line[1]-line[3];
#if 1
				const double len = (abs(dx)>abs(dy))? abs(dx) : abs(dy);
				const int kEdgeSamplingNum2 = static_cast <int> (len);
				dx/=len;
				dy/=len;
				/// TODO: adaptively change kEdgeSamplingNum
				for(int kk=0; kk<=kEdgeSamplingNum2; kk++){
					cv::Point p_tmp;
					p_tmp.x = static_cast <int> (line[2] + kk*dx);
					p_tmp.y = static_cast <int> (line[3] + kk*dy);
					if(0<=p_tmp.x&&p_tmp.x<img.cols&&0<=p_tmp.y&&p_tmp.y<img.rows){
						edge_points->push_back(p_tmp);
#ifdef DEBUG_LSD
						img_draw2.at<cv::Vec3b>(p_tmp.y,p_tmp.x) = cv::Vec3b(0,255,0);
#endif // DEBUG_LSD
					}
				}
#else
				const double len = sqrt(dx*dx+dy*dy);
				dx/=len;
				dy/=len;
				/// TODO: adaptively change kEdgeSamplingNum
				for(int k=0; k<kEdgeSamplingNum; k++){
					const double length = (len*k/kEdgeSamplingNum);
					cv::Point p_tmp;
					p_tmp.x = line[2] + length*dx;
					p_tmp.y = line[3] + length*dy;
					edge_points->push_back(p_tmp);
				}
#endif 
#ifdef DEBUG_LSD
				cv::line(img_draw, cv::Point(line[0],line[1]), cv::Point(line[2],line[3]),
					cv::Scalar(255*( (double)k/kLineSegmentNum), 0, 255));
#endif // DEBUG_LSD
			}
			edge_lists_.push_back(edge_points);
		}
#ifdef DEBUG_LSD
		cv::imshow("result_image_edge_cluster", img_draw);
		cv::imshow("result_image_edge_cluster_sampled", img_draw2);
#endif // DEBUG_LSD
	} // SECTION

	std::sort( edge_lists_.begin(), edge_lists_.end(), ComparePoint2iVecPtrPredicate );
	///cv::waitKey(-1);
#ifdef USE_OLD_LSD
 	delete lines;
#endif // USE_OLD_LSD
}

}// namespace
