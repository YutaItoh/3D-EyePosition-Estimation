/**
 @author Yuta Itoh <itoh@in.tum.de>, \n<a href="http://wwwnavab.in.tum.de/Main/YutaItoh">Homepage</a>.
**/

#include "iris_geometry.h"

namespace eye_tracker{
#ifdef DEBUG_IRIS_POSE
	Eigen::Matrix3Xd Ellipse::xy_;
	std::ofstream Ellipse::libus_pos_ofs("limbus_pos.txt");
	std::ofstream Ellipse::qe_ofs("qe.txt");
	std::ofstream Ellipse::offset_ofs("offset.txt");
	std::ofstream Ellipse::xy_ofs("xy.txt");
	std::ofstream Ellipse::k_ofs("k.txt");
#endif // DEBUG_IRIS_POSE
	bool Ellipse::SetExplicit( const double dx, const double dy, const double theta, const double a, const double b ){
		if( abs(a)<1e-15 || abs(b)<1e-15 ) return false;
			dx_ = dx;
			dy_ = dy;
			theta_ = theta;
			a_ = a;
			b_ = b;
			return ComputeEllipseParam(dx_,dy_,theta_,a_,b_);
	}

	bool Ellipse::ComputeEllipseParam( double dx, double dy, double theta, double a, double b){
		if( abs(a)<1e-15 || abs(b)<1e-15 ) return false;
		Eigen::Matrix3d A;
		Eigen::Matrix3d T;
		Eigen::Matrix3d T1;
		Eigen::Matrix3d T2;
		Eigen::Matrix3d S;
		const double c = cos(-theta-M_PI/2.0);
		const double s = sin(-theta-M_PI/2.0);
		T1 << 1.0, 0.0, -dx,
			  0.0, 1.0, -dy,
			  0.0, 0.0, 1.0;
		T2 << c,  -s, 0.0,
			  s,   c, 0.0,
			0.0, 0.0, 1.0;
		S << 1.0/(a*a), 0.0, 0.0,
			0.0, 1.0/(b*b), 0.0,
			0.0, 0.0, -1.0;
		T = T2*T1;
		A=T.transpose()*S*T;
///		A=A/A(2,2);
		A20_ = A(0,0)/A(2,2);
		A02_ = A(1,1)/A(2,2);
		A00_ = 1.0;
		A11_ = ( A(1,0)+A(0,1) )/A(2,2);
		A10_ = ( A(2,0)+A(0,2) )/A(2,2);
		A01_ = ( A(1,2)+A(2,1) )/A(2,2);
		//if(ComputeSixParameters(A20_,A11_,A02_,A10_,A01_,A00_)==false)
		//	std::cout<<"False"<<std::endl;
		return true;
	}
	
	/**
	* @function ComputeSixParameters
	* @brief Compute ellipse properties from its polynomial coefficients
	*  A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0
	*  [1] "General Equation of the Ellipse"
	*  http://www.juanrayces.com/EikonalTidbits/General%20equation%20of%20the%20ellipse.pdf
	*  [2] "Information About Ellipses"
	*  http://www.geometrictools.com/Documentation/InformationAboutEllipses.pdf
	*/
	bool Ellipse::ComputeSixParameters( double A20, double A11, double A02, double A10, double A01, double A00 ){
		const double A20plusA02 = A20 + A02;
		const double PositiveVal = sqrt( pow(A20-A02,2) + A11*A11 );

		if ( (A20plusA02 - PositiveVal) <= 1e-14 ) return false;

		const double TmpVal = 4*A20*A02 - A11*A11;
		if( TmpVal <= 1e-14 ) return false;
		double dx = dx_;
		double dy = dy_;

		dx = ( -2.0*A02*A10 + A11*A01 )/TmpVal;
		dy = ( -2.0*A20*A01 + A11*A10 )/TmpVal;
		double theta = atan2( A11, (A20-A02) )/2;
		const double mu = ( A20*dx*dx + A11*dx*dy + A02*dy*dy - A00);/// See [2]

		if( mu <= 1e-14 ) return false;
		double a = sqrt( 2.0*mu/( A20plusA02 - PositiveVal ) );
		double b = sqrt( 2.0*mu/( A20plusA02 + PositiveVal ) );
		
		if( a!=a || b!=b ){
			return false;
		}
		a_=a;
		b_=b;
		theta_ = theta;
		dx_=dx;
		dy_=dy;
		return true;
	}
	
	/**
	* @function Translate
	* @brief Translate ellipse and update each parameter
	*/
	void Ellipse::Translate( const double dx, const double dy ){
		dx_+=dx;
		dy_+=dy;
		SetExplicit(dx_,dy_,theta_,a_,b_);

#ifdef DEBUG_IRIS_POSE
		for( size_t k=0; k<xy_.cols(); k++ ){
			xy_(0,k) +=dx;
			xy_(1,k) +=dy;
		}
#endif // DEBUG_IRIS_POSE
	}


	/**
	* @function draw3Ddisk
	* @brief Draw a 3D on a given image
	* @param K Intrinsic matrix
	* @param G Disk normal vector
	* @param L Disk position
	* @param r Disk radius
	* @param img0 Image
	* @param color
	*/
	void draw3Ddisk(const Eigen::Matrix3d &K, const Eigen::Vector3d &G, const Eigen::Vector3d &L, const double r, cv::Mat &img0, const cv::Vec3b &color){
				Eigen::Vector3d r1;
				Eigen::Vector3d r2;
				Eigen::Vector3d r0;
				r1<<1.0,1.0,1.0;
				r1 = G.cross(r1);
				r1.normalize();
				r2 = G.cross(r1);
				r0 = G;
				const size_t N=500;
				Eigen::Vector3d xyz;
				for( int k=0; k<N; k++){
					xyz = r*cos(k/(double)N*M_PI*2.0)*r1 + 
						  r*sin(k/(double)N*M_PI*2.0)*r2 + L;
					xyz=K*xyz;
					const int y = static_cast<int> ( xyz(1)/xyz(2) );
					const int x = static_cast<int> ( xyz(0)/xyz(2) );
					if( 0<=x && x<img0.cols &&
						0<=y && y<img0.rows )
						img0.at<cv::Vec3b>(y,x)=color;
				}
				xyz=K*L;
				const int y = static_cast<int> ( xyz(1)/xyz(2) );
				const int x = static_cast<int> ( xyz(0)/xyz(2) );
				if( 0<=x && x<img0.cols &&
					0<=y && y<img0.rows ){
///						cv::circle(img0, cv::Point(x,y), 3,cv::Scalar(color[0],color[1],color[2]),2);
						img0.at<cv::Vec3b>(y,x)=cv::Vec3b(0,0,255);
				}

	}
	
	/**
	* @function ComputeEllipsePose
	* @brief Compute the 6DoF pose of an ellipse from the camera intrinsic parameter, the limbus radius
	*  and ellipse parameters (A20*x^2  + A11 * x^1 * y^1 + ... + A00 * x^0 * y^0 = 0)
	*  [1] Image-based Eye Pose and Reflection Analysis for Advanced Interaction Techniques and Scene Understanding
	*  http://www.lab.ime.cmc.osaka-u.ac.jp/paper/datas/2011/05/Nitschke_0406/Nitschke_201105_paper.pdf
	*  (dx, dy) is an offset in case ellipse param is estimated in a ROI image
	*/
	bool Ellipse::ComputeEllipsePose( const Eigen::Matrix3d &K, const double rL, cv::Mat &img,	
		std::vector<Eigen::Vector3d> &limbus_positions,
		std::vector<Eigen::Vector3d> &eye_positions,
		std::vector<Eigen::Vector3d> &eye_gazes){
		/// Convert the ellipse parameters from poynomial form to quadratic form
		Eigen::Matrix3d Qe0;
		Qe0 << A20_,     A11_*0.5, A10_*0.5,
			   A11_*0.5, A02_,     A01_*0.5,
			   A10_*0.5, A01_*0.5, A00_; /// 

#ifdef DEBUG_IRIS_POSE
		std::cout<<"K "<<std::endl<<K<<std::endl;
		std::cout<<"Qe0 "<<std::endl<<Qe0<<std::endl;
		qe_ofs<<std::setprecision(16)<<Qe0<<std::endl; 
		if(xy_.rows() == 3){
			Eigen::MatrixXd tmp=Qe0*xy_;
			tmp = xy_.array()*tmp.array();
			const double error_sum = tmp.sum();
			///std::cout<<"xy_ "<<std::endl<<xy_<<std::endl;
			xy_ofs <<std::setprecision(16)<<xy_<<std::endl;
			std::cout<<"error "<<error_sum<<std::endl;
		}
		std::cout<<"Qe0 "<<std::endl<<Qe0<<std::endl;
		std::cout<<"K "         <<std::endl<<K<<std::endl;
		k_ofs     <<std::setprecision(16)<<K<<std::endl;
#endif // DEBUG_IRIS_POSE

		// (14)
		Eigen::Matrix3d Qe;
		Qe = K.transpose() * Qe0 * K;
#ifdef DEBUG_IRIS_POSE
		std::cout<<"Kt*Qe0*K "<<std::endl<<Qe<<std::endl;
#endif // DEBUG_IRIS_POSE
		// (18) Decompose the general ellipse
		Eigen::EigenSolver<Eigen::Matrix3d> eigenSystem(Qe);
		const Eigen::Matrix3cd evecs = eigenSystem.eigenvectors();
		const Eigen::Vector3cd evals = eigenSystem.eigenvalues();
		/// No imaginary eigenvalues
		for( int row=0; row<evals.rows(); row++)
		{
			if(evals(row).imag() != 0.0) return false; /// ellipse is degenerated!
		}
		
		// (20) Sort the eigen values and vectors so that the condition in (20) is held
		const size_t kCombinationNum = 6;
		const size_t indices[kCombinationNum][3] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0} };
		int maximum_eval_index = 0;
		double a,b,c;
		Eigen::Matrix3d V;
		for( int k=0; k<kCombinationNum; k++){
			a = evals(indices[k][0]).real();
			b = evals(indices[k][1]).real();
			c = evals(indices[k][2]).real();
			if( a*b>0 && a*c<0 && abs(a)>=abs(b) ){ /// the condition for eigen valeus
				for( int kk=0; kk<3; kk++)
					V.col(kk) = evecs.col(indices[k][kk]).real();
				break;
			}
		}
		if( abs(a-c)<1e-16 || abs(a) < 1e-16 || abs(b) < 1e-16) return false; /// cannot be an ellipse
#ifdef DEBUG_IRIS_POSE
		std::cout<<"a "<<std::endl<<a<<std::endl;
		std::cout<<"b "<<std::endl<<b<<std::endl;
		std::cout<<"c "<<std::endl<<c<<std::endl;
#endif // DEBUG_IRIS_POSE
		
		// (22)
		const double g  = sqrt( (b-c)/(a-c) );
		const double h  = sqrt( (a-b)/(a-c) );
		const size_t kSignCombinationNum = 8;
		const int signs[kSignCombinationNum][3] = { {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1} };
		double rc;
		Eigen::Vector3d v1;
		Eigen::Vector3d v2;
		Eigen::Vector3d L;
		Eigen::Vector3d G;
		Eigen::Vector3d eyePosition;
#ifdef DEBUG_IRIS_POSE
		std::cout<<"K "<<std::endl<<K<<std::endl;
		std::cout<<"g "<<g<<std::endl;
		std::cout<<"h "<<h<<std::endl;
		cv::Mat img0 = img.clone();
		// GreenC
///		cv::rectangle(img0, cv::Point(dx,dy), cv::Point(dx+img0.rows, dy+img0.cols), cv::Scalar(0,200,0), 4, 8);
#endif // DEBUG_IRIS_POSE

		/// Init. output variables
		limbus_positions.resize(0);
		eye_positions.resize(0);
		eye_gazes.resize(0);
		for( int k=0; k<kSignCombinationNum; k++){ /// for each eye position candidate
			const double S1 = signs[k][0];
			const double S2 = signs[k][1];
			const double S3 = signs[k][2];
			rc = S3*sqrt(-a*c)/b;
			const double zc = rL/rc;
			v1 << S2*h*c/b, 0.0, -S1*g*a/b;
			v2 << S2*h,     0.0, -S1*g;
			const double d_LC = 5.53*0.001;
			const double d_CE = 5.70*0.001;
			const double D = ( d_LC + d_CE );/// distance from limbus center to the eye center
			L = zc*(V*v1); /// 3D position of the circler limbus
			G = (V*v2); /// normal vector of eye gaze
			eyePosition =  L - D*G;/// 3D position of the eye ball
			Eigen::Vector3d corneaPosition =  L - d_LC*G;/// 3D position of the eye ball

			/// Reject if the limbus position is behind the camera or the gace direction is not towards the camera
			if( L[2]>0.0 || G[2]<0.0 ) {
				continue;
			}
			//if( L[2]<0.0 || G[2]>0.0 ) continue;

///			std::cout<< "Cornea&Eye Position(Negative Z): " << std::endl << corneaPosition.transpose() << " "<< eyePosition.transpose() <<std::endl;

			limbus_positions.push_back(L);
			eye_positions.push_back(eyePosition);
			eye_gazes.push_back(G);
#ifdef DEBUG_IRIS_POSE
			std::cout<<std::endl;
			v1 = K*L;
			v2 = L+G*D;
			v2 = K*v2;

			/// Is limbus center L located in front of the camera?  Is the gaze vector G facing the camera?
			if( abs(v1(2)) > 1e-15 &&  abs(v1(2)) > 1e-15 )
			{
				static bool  hoge=true;
				/// Draw 3D limbus & 3D eyeballs
				{
					Eigen::Vector3d epos;
	
					const double rE =  sqrt(rL*rL+D*D);///12.60*0.001;/// Eye radius
					const double N=40;
					cv::Vec3b orange(0,102,255);
					cv::Vec3b green(0,255,0);
					cv::Vec3b red(0,0,255);
					
					for(int k=0;k<=N;k++){		
						cv::Vec3b orange(0,102,128*k/N);
						epos =  L - D*G - rE*G + rE*G*k/N;
						const double r = sqrt(rE*rE - pow((eyePosition-epos).norm(),2) );
						draw3Ddisk(K, G, epos, r, img0, orange);
					}
					for(int k=0;k<=N;k++){			
						cv::Vec3b orange(0,102,128*(1-k/N)+128);
						epos =   L - D*G*k/N;
						const double r = sqrt(rE*rE - pow((eyePosition-epos).norm(),2) );
						draw3Ddisk(K, G, epos, r, img0, orange);
					}
					draw3Ddisk(K, G, L, rL, img0, red);
					draw3Ddisk(K, G, eyePosition, rE, img0, green);
				}
				std::cout<<"rc "<<rc<<std::endl;
				std::cout<<"L "<<L.transpose()<<std::endl;
				std::cout<<"G "<<G.transpose()<<std::endl;
				std::cout<<"E "<<eyePosition.transpose()<<std::endl;
				std::cout<<"Gnorm "<<G.norm()<<std::endl;
				std::cout<<"S1="<<S1<< ", S2="<<S2<< ", S3="<<S3<<std::endl;
				v1/=v1(2);
				v2/=v2(2);
				cv::Point p1(v1(0),v1(1));
				cv::Point p2(v2(0),v2(1));
				cv::circle(img0, p1, 3,cv::Scalar(0,0,255),2);
				cv::line(img0, p1, p2, cv::Scalar(0,0,200), 2, 2);
				libus_pos_ofs<< eyePosition[0] << " " << eyePosition[1] << " " << eyePosition[2] << " " << G[0] << " " << G[1] << " " << G[2] << " " << L[0] << " " << L[1] << " " << L[2] << " " << rc <<std::endl;
			}
			for( int kk=0; kk<xy_.cols(); kk++){
				cv::circle(img0, cv::Point(xy_(0,kk),xy_(1,kk)), 2,cv::Scalar(0,0,255),2);
			}
			cv::circle(img0, cv::Point(dx_,dy_), 2,cv::Scalar(0,0,255),2);
#endif // DEBUG_IRIS_POSE
		} /// for each eye position candidate

#ifdef DEBUG_IRIS_POSE
		cv::imshow("IrisPose", img0);
#endif // DEBUG_IRIS_POSE
		return true;
	}

} /// namespace eye_tracker