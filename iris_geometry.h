#ifndef IRIS_GEOMETRY_H
#define IRIS_GEOMETRY_H

/**
 @author Yuta Itoh <itoh@in.tum.de>, \n<a href="http://wwwnavab.in.tum.de/Main/YutaItoh">Homepage</a>.
**/


#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>

//#include "tracker_log.h"

#if 1
  /// #define DEBUG_RANSAC
  #define DEBUG_IRIS_POSE
  ///#define DEBUG_FITELLIPSESUB
#endif

#ifdef DEBUG_IRIS_POSE
#include <fstream>
#endif // DEBUG_IRIS_POSE


namespace eye_tracker{
	
typedef std::vector<cv::Point2i> EdgeList;
typedef std::vector<cv::Point2f> EdgeListF;

/**
* @function calcPseudoColor
* @brief Compute Pseudo Color from a range parameter "phase" it takes [0,1]
*/
inline cv::Vec3b calcPseudoColor(double phase, double shift = 0.0)
{
    phase = std::max(std::min(phase,1.0), 0.0); // range [0...1]
    shift += M_PI+M_PI/4;     // [Blue ... Red]
    return cv::Vec3b
    (
        uchar( 255*(sin(1.5*M_PI*phase + shift + M_PI   ) + 1)/2.0 ),
        uchar( 255*(sin(1.5*M_PI*phase + shift + M_PI/2 ) + 1)/2.0 ), 
        uchar( 255*(sin(1.5*M_PI*phase + shift          ) + 1)/2.0 )
    );
} 

/**
* @class Ellipse
* @brief General ellipse using the 5 parameters.
*        A general ellipse can be written as:
*        A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0
*/
class Ellipse{
public:
	/// Initialize so that it represents a normal circle: x^2 + y^2 = 1
	Ellipse()
		: ///A20_(1.0), A11_(0.0), A02_(1.0), A10_(0.0), A01_(0.0), A00_(-1.0),
		  dx_(0.0), dy_(0.0), theta_(0.0), a_(1.0), b_(1.0)
	{
		SetExplicit(dx_,dy_,theta_,a_,b_);
	}
	~Ellipse(){};
	
	Ellipse& operator=(const Ellipse& rhs){
		A20_=rhs.A20_;
		A11_=rhs.A11_;
		A02_=rhs.A02_;
		A10_=rhs.A10_;
		A01_=rhs.A01_;
		A00_=rhs.A00_;
		dx_=rhs.dx_;
		dy_=rhs.dy_;
		theta_=rhs.theta_;
		a_=rhs.a_;
		b_=rhs.b_;
	}
	
	Ellipse(const Ellipse& other){
		A20_=other.A20_;
		A11_=other.A11_;
		A02_=other.A02_;
		A10_=other.A10_;
		A01_=other.A01_;
		A00_=other.A00_;
		dx_=other.dx_;
		dy_=other.dy_;
		theta_=other.theta_;
		a_=other.a_;
		b_=other.b_;
	}

	inline double x(){ return dx_;}
	inline double y(){ return dy_;}
	inline double a(){ return a_;}
	inline double b(){ return b_;}
	inline double t(){ return theta_;}
	inline double r(){ return b_/a_;}
	inline double BoundingArea(){ return 4.0*b_*a_;}
	inline void PrintExplicit(){
		std::cout<<  "a= " << a() <<std::endl;
		std::cout<<  "b= " << b() <<std::endl;
		std::cout<<  "t= " << t() <<std::endl;
		std::cout<<  "dx= " << x() <<std::endl;
		std::cout<<  "dy= " << y() <<std::endl;
	}
	inline void PrintPolynomial(){
		std::cout << "A20: " << A20_ <<std::endl;
		std::cout << "A02: " << A02_ <<std::endl;
		std::cout << "A11: " << A11_ <<std::endl;
		std::cout << "A10: " << A10_ <<std::endl;
		std::cout << "A01: " << A01_ <<std::endl;
		std::cout << "A00: " << A00_ <<std::endl;
	}

	bool SetExplicit( const double dx, const double dy, const double theta, const double a, const double b );
		
	/**
	* @function CenterSquaredDistance
	* @brief Distance between a given point and ellipse center
	*/
	inline double CenterSquaredDistance( cv::Point pt ){
		return pow(pt.x-dx_,2.0) + pow(pt.y-dy_,2.0);
	}
	/**
	* @function DrawEllipse
	* @brief Draw current ellipse 
	*/
	void DrawEllipse( cv::Mat img ){
		const double angle_deg = theta_/M_PI*180;
//		const cv::Scalar color = cv::Scalar(0,200,0);
		/// Choose Color
#if 1
		/// Color based on the distance between the iris center estimates
		const int iris_est_x = img.cols; /// Assume iris is at the center of eye
		const int iris_est_y = img.rows; ///
		const double d = sqrt( pow(dx_-iris_est_x,2) + pow(dy_-iris_est_y,2) );
		const double phase = abs(sin(d));
		const cv::Scalar color = calcPseudoColor(phase);
#else
		const double d = sqrt( pow(a_,2) + pow(b_,2) );/// Rough Ellipse size
		const cv::Scalar color = calcPseudoColor(d/100);
#endif
		///std::cout<<"ellipse diam: " << a_ << " " << b_ << std::endl;
		const int kThickness = 1;
		const cv::Point kCenter((int)dx_, (int)dy_); /// ellipse center
		const cv::Point kB( (int)(  b_* cos(theta_) ), 
			                (int)(  b_* sin(theta_) ) );/// axis1
		const cv::Point kA( (int)( -a_* sin(theta_) ), 
			                (int)( a_* cos(theta_) ) );/// axis2

		cv::ellipse(img, kCenter, cv::Size((int)b_, (int)a_), angle_deg, angle_deg, angle_deg+360, color, kThickness, 8);
		cv::circle(img, kCenter, 5, color, kThickness, 8);
		cv::line(img, kCenter, kCenter + kA , color, kThickness, 8);
		cv::line(img, kCenter, kCenter + kB , color, kThickness, 8);
	}

	/**
	* @function DrawEllipse
	* @brief Draw current ellipse 
	*/
	void DrawEllipseManual( cv::Mat img, double phase ){
		const double angle_deg = theta_/M_PI*180;
//		const cv::Scalar color = cv::Scalar(0,200,0);
		/// Choose Color
#if 1
		/// Color based on the distance between the iris center estimates
		const int iris_est_x = img.cols; /// Assume iris is at the center of eye
		const int iris_est_y = img.rows; ///
		const cv::Scalar color = calcPseudoColor(phase);
#else
		const double d = sqrt( pow(a_,2) + pow(b_,2) );/// Rough Ellipse size
		const cv::Scalar color = calcPseudoColor(d/100);
#endif
		///std::cout<<"ellipse diam: " << a_ << " " << b_ << std::endl;
		const int kThickness = 1;
		cv::ellipse(img, cv::Point((int)dx_, (int)dy_), cv::Size((int)b_, (int)a_), angle_deg, angle_deg, angle_deg+360, color, kThickness, 8);
		cv::circle(img, cv::Point((int)dx_, (int)dy_), 5, color, kThickness, 8);
	}

	void WarpEllipse( cv::Mat &img, cv::Mat &dst ){
		if( a_!=a_ || b_!=b_ ){
			return;
		}
		const double len = (a_>b_)? a_ : b_;
		const int w = (int)(2.0*len);
		const int h = (int)(2.0*len);
		dst = cv::Mat(h,w,CV_8UC3);
		const double c= cos(theta_);
		const double s= sin(theta_);
		const double kScale = 1.0;
		const double a= a_*kScale;
		const double b= b_*kScale;

		const cv::Point2f ellipse_pt[]={
                cv::Point2f( (float)(dx_ - c*a - s*b), (float)(dy_ - s*a + c*b) ), /// (-a, b)
                cv::Point2f( (float)(dx_ + c*a - s*b), (float)(dy_ + s*a + c*b) ), /// ( a, b)
                cv::Point2f( (float)(dx_ + c*a + s*b), (float)(dy_ + s*a - c*b) ), /// ( a,-b)
                cv::Point2f( (float)(dx_ - c*a + s*b), (float)(dy_ - s*a - c*b) ), /// (-a,-b)
		};
#if 0
		cv::line(img, ellipse_pt[0], ellipse_pt[1], cv::Scalar(0,0,200), 3, 4);
		cv::line(img, ellipse_pt[1], ellipse_pt[2], cv::Scalar(0,200,0), 3, 4);
		cv::line(img, ellipse_pt[2], ellipse_pt[3], cv::Scalar(200,0,0), 3, 4);
#endif
		const cv::Point2f circle_pt[]={
                cv::Point2f(0.0, 0.0),
                cv::Point2f(  (float)w, 0.0),
                cv::Point2f(  (float)w,   (float)h),
				cv::Point2f(0.0,   (float)h)
		};
		cv::Mat e2c_warp = cv::getPerspectiveTransform( ellipse_pt, circle_pt );
		cv::warpPerspective( img, dst, e2c_warp, dst.size() );
	}

	template <class T> int FitEllipse( const T &points1, const T &points2,
		const cv::Mat &src_dx, const cv::Mat &src_dy )
	{
		T points = points1;
		ConcatinatePoint2iVec( points, points2 );
		return FitEllipse( points, src_dx, src_dy );
	}
	
//	/**
//	* @function RANSAC
//	* @brief estimate Ellipse parameters by using RANSAC
//	*/
//template <class  T>
//	int RANSAC( const std::vector<cv::Point_<T> >&points, 
//		const cv::Mat &src_dx, const cv::Mat &src_dy );
//
//	/**
//	* @function CountInliers
//	* @brief Count the number of inliers among given 2Dpoints
//	*/
//template <class T>
//	double  CountInliers( const std::vector<T> &points, int &inlier_count, std::vector<int> &inlier_indice,
//		const cv::Mat &src_dx, const cv::Mat &src_dy );
//	
#ifdef DEBUG_RANSAC
	cv::Mat tmp;
	void SetDebugMat( const cv::Mat &tmp2){
		tmp=tmp2.clone();
	}
#endif // DEBUG_RANSAC

	/**
	* @function RANSAC
	* @brief estimate Ellipse parameters by using RANSAC
	*/
template <class  T>
	int RANSAC( const std::vector<cv::Point_<T> >&points, 
		const cv::Mat &src_dx, const cv::Mat &src_dy )
	{
		const size_t kNum = 5; /// minimum training set sample num
		const size_t kSrcNum = points.size();
		const size_t kEarlyTerminationNum = static_cast<size_t>( 0.95*kSrcNum );
		if( kSrcNum < kNum) return false;

		/// Decide max iteration # so that it achieves a certain success rate
		const double kInlierProb = 0.6; /// Inlier probability
		const double kTargetProb = 0.001; /// Expected failure probablity
		const int kMaxIteration = (int)( log(kTargetProb)/log(1.0-pow(kInlierProb,(int)kNum)) );

		int idxs[kNum];
		std::vector<cv::Point_<T> > min_points(kNum);
		size_t best_inlier_count = 0;
		double best_score = 0.0;
		std::vector<int> best_inlier_indice;

#ifdef DEBUG_RANSAC
		cv::Mat tmp0=tmp.clone();
#endif // DEBUG_RANSAC
		for( int itr=0; itr<kMaxIteration; itr++){
			/// Chose minimum number of samples
			for( int i=0; i<kNum; i++ ){
				bool is_duplicated = false;
				do 
				{
					is_duplicated = false;
					int idx	= rand() % kSrcNum;
					idxs[i] = idx;
					for( int k=0; k<i; k++ ){
						if( idxs[k] == idx ) is_duplicated = true;
					}
				} while( is_duplicated );
				min_points[i] = points[ idxs[i] ];
			}
			/// Compute Ellipse parameter
			const bool is_estimation_success = FitEllipseSub( min_points );

#ifdef DEBUG_RANSAC
			///			tmp=cv::Scalar(0);
			for( int i=0; i<kNum; i++ ){
				const int yy = (int)min_points[ i ].y;
				const int xx = (int)min_points[ i ].x;
				if( xx<0||xx>=tmp.cols||yy<0||yy>=tmp.rows)continue;
				tmp.at<cv::Vec3b>(yy,xx) = cv::Vec3b(0,0,255);
			}
#endif // DEBUG_RANSAC
			if( is_estimation_success ==false )
				continue;

			/// Count inliers
			size_t inlier_count;
			std::vector<int> inlier_indice;
			///				const double current_score;
			CountInliers(points, inlier_count, inlier_indice, src_dx, src_dy);
			/// if it gives largest number of inliers
			if( best_inlier_count < inlier_count){
				///				if( best_score < current_score){
				/// remember the new inliers
				///					best_score = current_score;
				best_inlier_count  = inlier_count;
				best_inlier_indice = inlier_indice;
#ifdef DEBUG_RANSAC
				DrawEllipse(tmp);
				for( size_t i=0; i<kSrcNum; i++ ){
					const int yy = (int)points[ i ].y;
					const int xx = (int)points[ i ].x;
					if( xx<0||xx>=tmp.cols||yy<0||yy>=tmp.rows)continue;
					tmp.at<cv::Vec3b>(yy,xx) = cv::Vec3b(128,128,128);
				}
				cv::imshow("RANSAC",tmp);
				//cv::waitKey(-1);
				std::cout<< "best_inlier_count:" << best_inlier_count << std::endl;
#endif // DEBUG_RANSAC
			}



			// Early termination for  95% inliers			
			if(best_inlier_count>=kEarlyTerminationNum) break;
		}

		/// Using the largest set, compute the final Ellipse parameter
		std::vector<cv::Point_<T> > best_points(best_inlier_indice.size());
		for( size_t i=0; i<best_inlier_indice.size(); i++ ){
			best_points[i]=points[ best_inlier_indice[i] ];
		}
#ifdef DEBUG_RANSAC

		tmp=tmp0.clone();
		FitEllipseSub( best_points );
		tmp0=cv::Scalar(0);
		for( int i=0; i<points.size(); i++ ){
			const int yy = (int)points[ i ].y;
			const int xx = (int)points[ i ].x;
			if( xx<0||xx>=tmp0.cols||yy<0||yy>=tmp0.rows)continue;
			tmp0.at<cv::Vec3b>(yy,xx) = cv::Vec3b(128,128,128);
		}
		for( int i=0; i<best_inlier_indice.size(); i++ ){
			const int yy = (int)points[ best_inlier_indice[i] ].y;
			const int xx = (int)points[ best_inlier_indice[i] ].x;
			if( xx<0||xx>=tmp0.cols||yy<0||yy>=tmp0.rows)continue;
			tmp0.at<cv::Vec3b>(yy,xx) = cv::Vec3b(0,0,255);
		}
		DrawEllipse(tmp0);
		cv::imshow("BestRANSAC",tmp0);
		cv::waitKey(-1);	
#endif // DEBUG_RANSAC
		
#ifdef DEBUG_IRIS_POSE
		xy_.resize(3,best_points.size());
		for( size_t k=0; k<best_points.size(); k++ ){
			xy_(0,k) = best_points[k].x;
			xy_(1,k) = best_points[k].y;
			xy_(2,k) = 1.0;
		}
#endif // DEBUG_IRIS_POSE
		if( FitEllipseSub( best_points ) ) return best_inlier_count;

		return 0;
	}
	
	/**
	* @function CountInliers
	* @brief Count the number of inliers among given 2Dpoints
	*/
template <class T>
	double  CountInliers( const std::vector<T> &points, size_t &inlier_count, std::vector<int> &inlier_indice,
		const cv::Mat &src_dx, const cv::Mat &src_dy ){
		double score=0.0;
///		const double kInlierThreashold= pow(1.5,2);
		const double kThresh = (src_dx.rows+src_dx.cols)/2/50; /// heuristic
		const double kInlierThreashold= pow(kThresh,2);
		inlier_count=0;
		inlier_indice.clear();
		double dx0, dy0;
		double dx, dy;
#ifdef DEBUG_RANSAC
		cv::Mat tmp;
		cv::cvtColor(src_dx,tmp,CV_GRAY2BGR);
#endif // DEBUG_RANSAC
		for( size_t i=0; i<points.size(); i++ ){
			if( points[i].x<0 || points[i].x>=src_dx.cols ||
				points[i].y<0 || points[i].y>=src_dy.rows ) continue;
#if 0 /// DistanceSquared is very slow, instead we use approximated distance 
			const double d = DistanceSquared(points[i].x,points[i].y);
#else
			const double d = DistanceSimpleSquared(points[i].x,points[i].y);
#endif

			bool is_inlier = false;
			is_inlier = d < kInlierThreashold;

#if 1 /// with a gradient filtering assuming iris color is darker than sclera
			if ( is_inlier ){
				dx0 = src_dx.at<float>( (int)points[i].y,(int)points[i].x);
				dy0 = src_dy.at<float>( (int)points[i].y,(int)points[i].x);/// ToDo subsampling?
				ComputeEllipseGradient(points[i].x,points[i].y, dx, dy);
				score = (dx0*dx + dy0*dy)/( sqrt(dx*dx + dy*dy) * sqrt(dx0*dx0 + dy0*dy0) );
#ifdef DEBUG_RANSAC
				double d  = sqrt(dx*dx + dy*dy);
				double d0 = sqrt(dx0*dx0 + dy0*dy0);
				const double kGradScale =10.0;
				cv::line(tmp, 
					cv::Point(points[i].x, points[i].y),
					cv::Point(points[i].x+dx0/d0*kGradScale, points[i].y+dy0/d0*kGradScale),
					cv::Scalar(0,0,255), 1, CV_AA);
				cv::line(tmp, 
					cv::Point(points[i].x, points[i].y),
					cv::Point(points[i].x+dx/d*kGradScale, points[i].y+dy/d*kGradScale),
					cv::Scalar(0,255,0), 1, CV_AA);
				std::cout << "score " << acos(score)/M_PI*180.0<< std::endl;
#endif // DEBUG_RANSAC
				is_inlier = acos(score) < 10.0/180.0*M_PI;
			}
#endif /// without a gradient filtering


			if ( is_inlier ){
				inlier_count++;
				inlier_indice.push_back(i);
			}
		}
#ifdef DEBUG_RANSAC
		cv::imshow("Ellipse Gradient",tmp);
#endif // DEBUG_RANSAC
		if(inlier_count==0) return 0.0;
		return inlier_count;
		return score/(double)inlier_count;
	};
	
	/**
	* @function DistanceSimpleSquared
	* @brief Calculate rough distance between a 2D point and the ellipse 
	* The distance is calculated through the normalized circle world
	*/
	template <class T>
	double DistanceSimpleSquared( const T x0, const T y0 ){
		double x =  (x0-dx_)*cos(theta_) + (y0-dy_)*sin(theta_);
		double y = -(x0-dx_)*sin(theta_) + (y0-dy_)*cos(theta_);
		//		const double rab = sqrt( pow((x/a_),2.0) + pow((y/b_),2.0) );
		//		const double rba = sqrt( pow((x/b_),2.0) + pow((y/a_),2.0) );
		return (x*x+y*y)*pow( ( 1.0 - 1.0/( pow(x/b_,2.0) + pow(y/a_,2.0) ) ), 2.0 );
	}

	/**
	* @function DistanceSquared
	* @brief Calculate distance between a 2D point and the ellipse
	*/
	template <class T>
	double DistanceSquared( const T x0, const T y0 ){
		double x =  (x0-dx_)*cos(theta_) + (y0-dy_)*sin(theta_);
		double y = -(x0-dx_)*sin(theta_) + (y0-dy_)*cos(theta_);
		x = abs(x);
		y = abs(y);
		const double t0 = -pow(a_,2)*pow(y,2)-pow(b_,2)*pow(x,2)+pow(a_,2)*pow(b_,2);
		const double t1 = (-2*a_*b_*pow(y,2)-2*a_*b_*pow(x,2)+2*a_*pow(b_,3)+2*pow(a_,3)*b_);
		const double t2 = (-pow(b_,2)*pow(y,2)-pow(a_,2)*pow(x,2)+pow(b_,4)+4*pow(a_,2)*pow(b_,2)+pow(a_,4));
		const double t3 = (2*a_*pow(b_,3)+2*pow(a_,3)*b_);
		const double t4 = pow(a_,2)*pow(b_,2);
		const double a = t3/t4;
		const double b = t2/t4;
		const double c = t1/t4;
		const double d = t0/t4;
		const size_t kRootNum = 4;
		double cc[kRootNum+1]={d,c,b,a,1.0};
        cv::Mat coeffs(1, kRootNum+1, CV_64F, cc);
		std::vector< cv::Complex<double> > roots;
		cv::solvePoly(coeffs,roots);
#if 0
		std::cout << std::endl;
		std::cout << "a="<<a<<", b="<<b<<", c="<<c<<", d="<<d<< std::endl;
		std::cout<<"coeff="<<coeffs<<std::endl;
///		std::cout << "roots=    "<<roots[0] <<" "<< roots[1] << " " << roots[2] << " " << roots[3] <<std::endl;
		std::cout << "roots_re= "<<roots[0].re <<" "<< roots[1].re << " " << roots[2].re << " " << roots[3].re <<std::endl;
		std::cout << "roots_im= "<<roots[0].im <<" "<< roots[1].im << " " << roots[2].im << " " << roots[3].im <<std::endl;
#endif
		std::vector<double> roots_re;
		for (size_t k=0; k< kRootNum; ++k){
			if(abs(roots[k].im)<1e-13)
				roots_re.push_back(roots[k].re);
		}

		if ( roots_re.size() == 0 ) return DBL_MAX;
		double root = roots_re[0];
		for (size_t i(1); i < roots_re.size(); ++i)
			if( abs(root)>abs(roots_re[i]) ) root = roots_re[i];
		const double phi = atan2( (y*(a_+b_*root)),  (x*(b_+a_*root)) );
		const double dis = root*root*( pow(a_,2)*pow(cos(phi),2) + pow(b_,2)*pow(sin(phi),2) );
///		std::cout << "DistanceSquared, dis"<<dis<<std::endl;
		return dis;
	}

	
template <class T>
	int FitEllipse( const std::vector<cv::Point_<T> > &points,
		const cv::Mat &src_dx, const cv::Mat &src_dy )
	{
		return RANSAC(points, src_dx, src_dy);
	}

	/**
	* @function FitEllipseSub
	* @brief Fit an ellipse to given 2D points
	*     A C++ implementation of 
	*     "NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES"
	*     http://autotrace.sourceforge.net/WSCG98.pdf
	*     This code is modified from its C# cide implemented by Srikanth Kotagiri who uses Microsoft Public License (Ms-PL)
	*     http://skotagiri.wordpress.com/2010/06/19/c-implementation-for-fitting-an-ellipse-for-a-set-of-points/
	*/
template <class T>
	bool FitEllipseSub( const std::vector<cv::Point_<T> > &points )
	{
		const size_t numPoints = points.size();
		if(numPoints == 0) return false;
		Eigen::VectorXd X(numPoints,1);
		Eigen::VectorXd Y(numPoints,1);
		Eigen::MatrixXd D1(numPoints,3);
		Eigen::MatrixXd D2(numPoints,3);
		Eigen::Matrix3d S1;
		Eigen::Matrix3d S2;
		Eigen::Matrix3d S3;
		Eigen::Matrix3d T;
		Eigen::Matrix3d M;
		Eigen::Matrix3d C1;
		C1 << 0.0,  0.0, 0.5,
	     	  0.0, -1.0, 0.0,
			  0.5,  0.0, 0.0;
		Eigen::Vector3d a1;
		Eigen::Vector3d a2;
		Eigen::VectorXd result(6);
		//2 D1 = [x .? 2, x .* y, y .? 2]; % quadratic part of the design matrix
		//3 D2 = [x, y, ones(size(x))]; % linear part of the design matrix

		/// Normalize input
		/// Direct Least Square Fitting of Ellipses
		/// http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PILU1/
		for (size_t xx = 0; xx < points.size(); xx++)
		{
			const cv::Point2i &p = points[xx];
			X[xx] = p.x;
			Y[xx] = p.y;
		}

		const double mx = X.mean();
		const double my = Y.mean();
		const double sx = ( X.maxCoeff() - X.minCoeff() )/2;
		const double sy = ( Y.maxCoeff() - Y.minCoeff() )/2;

#ifdef DEBUG_FITELLIPSESUB
		std::cout<<"Observation"<<std::endl;
		std::cout<<"A=["<<std::endl;
#endif

		for (size_t xx = 0; xx < points.size(); xx++)
		{
			///		const cv::Point2i &p = points[xx];
			const double px = (X[xx]-mx)/sx;
			const double py = (Y[xx]-my)/sy;

			std::cout.precision(16);
#ifdef DEBUG_FITELLIPSESUB
			std::cout<<  px <<" "<< py <<";"<<std::endl;
#endif
			D1(xx, 0) = px * px;
			D1(xx, 1) = px * py;
			D1(xx, 2) = py * py;
			D2(xx, 0) = px;
			D2(xx, 1) = py;
			D2(xx, 2) = 1;
		}

#ifdef DEBUG_FITELLIPSESUB
		std::cout<<"]"<<std::endl;
#endif
		//4 S1 = D1f * D1; % quadratic part of the scatter matrix
		S1 = D1.transpose() * D1;

		//5 S2 = D1f * D2; % combined part of the scatter matrix
		S2 = D1.transpose() * D2;

		//6 S3 = D2f * D2; % linear part of the scatter matrix
		S3 = D2.transpose() * D2;

		//7 T = - inv(S3) * S2f; % for getting a2 from a1

		Eigen:: FullPivLU<Eigen::Matrix3d>  lu(S3);
		lu.setThreshold(1e-8);
		if( lu.rank() < 3 ) return false;
		T = - lu.solve(S2.transpose());
		///	T = - S3.inverse() * S2.transpose();

		//8 M = S1 + S2 * T; % reduced scatter matrix
		M = S1 + (S2 * T);


#ifdef DEBUG_FITELLIPSESUB
		std::cout<< "M S1 + S2T"         << std::endl << M <<std::endl;
		Eigen::Matrix3d TMP=S2 * S3.inverse() * S2.transpose();
		std::cout<< "M S1"         << std::endl <<S1 <<std::endl;
		std::cout<< "M      S2S3invS2T"         << std::endl <<TMP <<std::endl;
		std::cout<< "M S1 - S2S3invS2T"         << std::endl << S1-TMP <<std::endl;
		///	std::cout<< "M S1S2T_2"         << std::endl << S1 -() <<std::endl;
#endif
		//9 M = (M(3, <img src="http://s0.wp.com/wp-includes/images/smilies/icon_smile.gif?m=1129645325g" alt=":)" class="wp-smiley">  ./ 2; - M(2, <img src="http://s0.wp.com/wp-includes/images/smilies/icon_smile.gif?m=1129645325g" alt=":)" class="wp-smiley"> ; M(1, <img src="http://s0.wp.com/wp-includes/images/smilies/icon_smile.gif?m=1129645325g" alt=":)" class="wp-smiley">  ./ 2]; % premultiply by inv(C1)
		M = C1 * M;

#ifdef DEBUG_FITELLIPSESUB
		//	std::cout<< "D1" << std::endl << D1 <<std::endl;
		///	std::cout<< "D1" << std::endl << D2 <<std::endl;
		std::cout<< "S1" << std::endl << S1 <<std::endl;
		std::cout<< "S2" << std::endl << S2 <<std::endl;
		std::cout<< "S3" << std::endl << S3 <<std::endl;
		std::cout<< "T " << std::endl <<  T <<std::endl;
		std::cout<< "C1M"         << std::endl << M <<std::endl;
#endif

		//10 [evec, eval] = eig(M); % solve eigensystem
		const Eigen::Matrix3d M2 = M.cast<double>();
		Eigen::EigenSolver<Eigen::Matrix3d> eigenSystem(M2);

		//11 cond = 4 * evec(1, <img src="http://s0.wp.com/wp-includes/images/smilies/icon_smile.gif?m=1129645325g" alt=":)" class="wp-smiley">  .* evec(3, <img src="http://s0.wp.com/wp-includes/images/smilies/icon_smile.gif?m=1129645325g" alt=":)" class="wp-smiley">  - evec(2, <img src="http://s0.wp.com/wp-includes/images/smilies/icon_smile.gif?m=1129645325g" alt=":)" class="wp-smiley">  .? 2; % evaluate afCa
		//12 a1 = evec(:, find(cond > 0)); % eigenvector for min. pos. eigenvalue
		const Eigen::Matrix3Xcd evecs = eigenSystem.eigenvectors();
		const Eigen::Vector3cd  evals = eigenSystem.eigenvalues();
		int maximum_eval_index = 0;
		for( int row=0; row<evals.rows(); row++){
			/// TODO: check this condition
			int col=row;
			std::complex<double> condition = 4.0*evecs(0,col)*evecs(2,col) - evecs(1,col)*evecs(1,col);
			///		double condition = evecs(0,row).real() * evecs(2,row).real() - evecs(1,row).real() * evecs(1,row).real();
			if( condition.imag() == 0 && condition.real() > 0 ){
			///if( evals[row].imag() == 0 && evals[row].real() > 0 ){
				maximum_eval_index = row;
				break;
			}
		}
		a1 = evecs.col(maximum_eval_index).real().cast<double>();

		//13 a2 = T * a1; % ellipse coefficients
		a2 = T * a1;

		//14 a = [a1; a2]; % ellipse coefficients
		// a = [ A B C D E F ]
		// A x2 + Bxy + Cy2 + Dx + Ey + F = 0
		/// Unnormalize
		result[0] = a1(0, 0)*sy*sy;
		result[1] = a1(1, 0)*sx*sy;
		result[2] = a1(2, 0)*sx*sx;
		result[3] = a2(0, 0)*sx*sy*sy -2.0f*mx*result[0]- my*result[1];
		result[4] = a2(1, 0)*sx*sx*sy                       -mx*result[1] - 2.0f*my*result[2];
		result[5] = a2(2, 0)*sx*sx*sy*sy +     result[0]*mx*mx + result[1]*mx*my + result[2]*my*my  - a2(0, 0)*sx*sy*sy*mx - a2(1, 0)*sx*sx*sy*my;


#ifdef DEBUG_FITELLIPSESUB
		std::cout<< "M"         << std::endl << M2 <<std::endl;
		std::cout<< "Eigen Values"   << std::endl << evals <<std::endl;
		std::cout<< "Eigen Vectors"  << std::endl << evecs <<std::endl;
		std::cout<< "Chosen Eig Vec "<< std::endl << a1 <<std::endl;
		std::cout<< "Ellipse param." <<std::endl;
		std::cout<< result <<std::endl;
#endif
		// a = [ A B C D E F ]
		// A x2 + Bxy + Cy2 + Dx + Ey + F = 0
		/// General Equation of the Ellipse
		/// http://www.juanrayces.com/EikonalTidbits/General%20equation%20of%20the%20ellipse.pdf]
#if 1
		A20_ = result[0]/result[5];// x^2y^0
		A11_ = result[1]/result[5];// x^1y^1
		A02_ = result[2]/result[5];// x^0y^2
		A10_ = result[3]/result[5];// x^1y^0
		A01_ = result[4]/result[5];// x^0y^1
		A00_ = result[5]/result[5];// x^0y^0
#else
		A20_ = result[0];// x^2y^0
		A11_ = result[1];// x^1y^1
		A02_ = result[2];// x^0y^2
		A10_ = result[3];// x^1y^0
		A01_ = result[4];// x^0y^1
		A00_ = result[5];// x^0y^0
#endif

#ifdef DEBUG_FITELLIPSESUB
		std::cout<< "Result"<<std::endl;
		for(int i=0;i<6;i++)std::cout<< result[i]/result[5]<<std::endl;
#endif
		return ComputeSixParameters(A20_,A11_,A02_,A10_,A01_,A00_);
	}

	/**
	* @function ComputeEllipsePose
	* @brief Compute the 6DoF pose of an ellipse from the camera intrinsic parameter, the limbus radius
	*  and ellipse parameters (A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0)
	*  [1] Image-based Eye Pose and Reflection Analysis for Advanced Interaction Techniques and Scene Understanding
	*  http://www.lab.ime.cmc.osaka-u.ac.jp/paper/datas/2011/05/Nitschke_0406/Nitschke_201105_paper.pdf
	*  (dx, dy) is an offset in case ellipse param is estimated in a ROI image
	*/
	bool ComputeEllipsePose( const Eigen::Matrix3d &K, const double rL, cv::Mat &img,	
		std::vector<Eigen::Vector3d> &limbus_positions,
		std::vector<Eigen::Vector3d> &eye_positions,
		std::vector<Eigen::Vector3d> &eye_gazes);
	
	/**
	* @function Translate
	* @brief Translate ellipse and update each parameter
	*/
	void Translate( const double dx, const double dy );
	
private:

	 template <class T> inline void ConcatinatePoint2iVec( std::vector<T> &head, const std::vector<T> &tail){
		for( size_t i=0; i<tail.size(); i++ ){
			head.push_back(tail[i]);
		}
	}

	/**
	* @function ComputeEllipseGradient
	* @brief Compute ellipse gradient vector at a point;
	*  A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0
	*/
	void ComputeEllipseGradient( const double x, const double y, double &dx , double &dy){
		dx = 2.0*A20_*x + A11_*y + A10_;
		dy = 2.0*A02_*y + A11_*x + A01_;
	}
	
	/**
	* @function ComputeEllipseParam
	* @brief Compute ellipse polynomial properties from its geometric parameters
	*  A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0
	*/
	bool ComputeEllipseParam( double dx, double dy, double theta, double a, double b);
	
	/**
	* @function ComputeSixParameters
	* @brief Compute ellipse properties from its polynomial coefficients
	*  A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0
	*  [1] "General Equation of the Ellipse"
	*  http://www.juanrayces.com/EikonalTidbits/General%20equation%20of%20the%20ellipse.pdf
	*  [2] "Information About Ellipses"
	*  http://www.geometrictools.com/Documentation/InformationAboutEllipses.pdf
	*/
	bool ComputeSixParameters( double A20, double A11, double A02, double A10, double A01, double A00 );
	
	

	/// A general ellipse can be written as:
	/// A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0
	double A20_, A11_, A02_, A10_, A01_, A00_;
	double dx_, dy_; // Ellipse center
	double theta_; // Ellipse angle
	double a_, b_; // Elipse diameters
	///double r_; // Elipse diameters ratio 
#ifdef DEBUG_IRIS_POSE
	static Eigen::Matrix3Xd xy_;
	static std::ofstream qe_ofs;
	static std::ofstream offset_ofs;
	static std::ofstream xy_ofs;
	static std::ofstream k_ofs;
	static std::ofstream libus_pos_ofs;
#endif // DEBUG_IRIS_POSE

};

} /// namespace eye_tracker
#endif // IRIS_GEOMETRY_H