#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

double EPSILON = 0.1;

float euclideanDist(Point2f& p, Point2f& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

Mat LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                   Matx34d P,       //camera 1 matrix
                   Point3d u1,      //homogenous image point in 2nd camera
                   Matx34d P1       //camera 2 matrix
                                   )
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
              );
    Mat B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
                      -(u.y*P(2,3)  -P(1,3)),
                      -(u1.x*P1(2,3)    -P1(0,3)),
                      -(u1.y*P1(2,3)    -P1(1,3)));
 
    Mat X;
    solve(A,B,X,DECOMP_SVD);
 
    return X;
}

Mat_<double> IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
                                            Matx34d P,          //camera 1 matrix
                                            Point3d u1,         //homogenous image point in 2nd camera
                                            Matx34d P1          //camera 2 matrix
                                            ) {
    double wi = 1, wi1 = 1;
    Mat_<double> X(4,1); 
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
         
        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
         
        //breaking point
        if(abs(wi - p2x) <= EPSILON && abs(wi1 - p2x1) <= EPSILON) break;
         
        wi = p2x;
        wi1 = p2x1;
         
        //reweight equations and solve
        Matx43d A((u.x*P(2,0)-P(0,0))/wi,       (u.x*P(2,1)-P(0,1))/wi,         (u.x*P(2,2)-P(0,2))/wi,     
                  (u.y*P(2,0)-P(1,0))/wi,       (u.y*P(2,1)-P(1,1))/wi,         (u.y*P(2,2)-P(1,2))/wi,     
                  (u1.x*P1(2,0)-P1(0,0))/wi1,   (u1.x*P1(2,1)-P1(0,1))/wi1,     (u1.x*P1(2,2)-P1(0,2))/wi1, 
                  (u1.y*P1(2,0)-P1(1,0))/wi1,   (u1.y*P1(2,1)-P1(1,1))/wi1,     (u1.y*P1(2,2)-P1(1,2))/wi1
                  );
        Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3))/wi,
                          -(u.y*P(2,3)  -P(1,3))/wi,
                          -(u1.x*P1(2,3)    -P1(0,3))/wi1,
                          -(u1.y*P1(2,3)    -P1(1,3))/wi1
                          );
         
        solve(A,B,X_,DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
    }
    return X;
}

void TriangulatePoints(const vector<Point2f>& pt_set1,
                       const vector<Point2f>& pt_set2,
                       const Mat& Kinv,
                       const Matx34d& P,
                       const Matx34d& P1,
                       vector<Point3d>& pointcloud,
                       vector<Point2f>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
    vector depths;
#endif
 
    pointcloud.clear();
    correspImg1Pt.clear();
 
    cout << "Triangulating...";
    double t = getTickCount();
    unsigned int pts_size = pt_set1.size();
    for (unsigned int i=0; i<pts_size;i++) {
    	Point2f kp = pt_set1[i];
        Point3d u(kp.x,kp.y,1.0);
        Mat um = Kinv * Mat(u);
        // u = um.at(0);
        u = Point3d(um);
        Point2f kp1 = pt_set2[i];
        Point3d u1(kp1.x,kp1.y,1.0);
        Mat um1 = Kinv * Mat(u1);
        // u1 = um1.at(0);
        u1 = Point3d(um1);
 
        Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);
 

        {
            pointcloud.push_back(Point3d(X(0),X(1),X(2)));
            correspImg1Pt.push_back(pt_set1[i]);
        }
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Done. (";
}

// Mat getFundMat(vector<Point2f> pnt1, vector<Point2f> pnt2) {
// 	// for(int i=0;i<)
// }

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main() {
	// VideoCapture cap(0);
	// if(!cap.isOpened()) {
	// 	cout<<"Unable to open camera";
	// 	exit(-1);
	// }
	VideoCapture cap("Dataset/vid1.mp4");
	if( !cap.isOpened() )
        cout<< "Error when reading steam_avi"<<endl;

	int FPS = 24, ratio = 2;
	namedWindow( "tp", WINDOW_AUTOSIZE );
	namedWindow( "tp2", WINDOW_AUTOSIZE );
	Mat image1, image2;
	Mat gray1, gray2;

    Mat featureEx1, featureEx2;

	// cap>>image;
	// cout<<"rows: "<<image.rows<<" cols: "<<image.cols<<" ch: "<<image.channels()<<" "<<image.isContinuous()<<endl;

	while(true) {
		//Read images
		// image1 = imread("Dataset/im1.jpg");
		// image2 = imread("Dataset/im2.jpg");
		cap.read(image1);
		if(image1.empty()) {
			cout<<"e"<<endl;
			continue;
		}
		int n=10;
		while(n--) {
			cap>>image2;
		}
		cout<<image1.type()<<endl;

		//Convert to gray for SIFT
		cvtColor(image1, gray1, COLOR_BGR2GRAY);
		cvtColor(image2, gray2, COLOR_BGR2GRAY);

		//SIFT Stuff
		// SiftFeatureDetector detector1, detector2;
		vector<KeyPoint> keypoints1, keypoints2;
		// cout<<"detecting keypoints"<<endl;
		// detector1.detect(gray1, keypoints1);
		// detector2.detect(gray2, keypoints2);

		// SiftDescriptorExtractor sfe;

		// sfe.compute(gray1, keypoints1, featureEx1);
		// sfe.compute(gray2, keypoints2, featureEx2);

		std::vector<DMatch>  dmv, dmv_sel;
		// BFMatcher matcher(NORM_L2, true);
		// matcher.match(featureEx1, featureEx2, dmv);



		cv::Ptr<cv::FeatureDetector> detector;
		detector = cv::FeatureDetector::create("ORB");
		detector->detect(gray1, keypoints1);
    	detector->detect(gray2, keypoints2);

    	cv::Ptr<cv::DescriptorExtractor> extractor;
    	extractor = cv::DescriptorExtractor::create("ORB");
    	extractor->compute(gray1, keypoints1, featureEx1);
    	extractor->compute(gray2, keypoints2, featureEx2);

    	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    	matcher->match(featureEx1, featureEx2, dmv);


		Mat outp, img_matches;
		drawKeypoints(gray1, keypoints1, outp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


		imshow("tp", outp);

		vector<Point2f> pnt1, pnt2;

		//Print match array
		for(int i=0;i<dmv.size();i++) {
			// cout<<dmv[i].queryIdx<<" "<<dmv[i].trainIdx<<endl;
			cout<<dmv[i].queryIdx<<": "<<keypoints1[dmv[i].queryIdx].pt<<" "<<dmv[i].trainIdx<<": "<<keypoints2[dmv[i].trainIdx].pt<<endl;
			float dist = euclideanDist(keypoints1[dmv[i].queryIdx].pt, keypoints2[dmv[i].trainIdx].pt);
			cout<<"Dist: "<<dist<<endl;
			if(dist <= 70) {
				pnt1.push_back(keypoints1[dmv[i].queryIdx].pt);
				pnt2.push_back(keypoints2[dmv[i].trainIdx].pt);
				dmv_sel.push_back(dmv[i]);
			}
		}


		//Draw the matches in 2 images
		drawMatches(gray1, keypoints1, gray2, keypoints2, dmv_sel, img_matches);

		cout<<keypoints1.size()<<" "<<keypoints2.size()<<endl;

		imshow("tp2", gray1);
		imshow("tp2", img_matches);

		Mat K = (Mat_<float>(3,3) << 3288.1f, 0.0f, 1593.6f,
									0.0f, 3302.0f, 2097.6f,
									0.0f, 0.0f, 1.0f);

		// Mat K = (Mat_<float>(3,3) << 3.6f*75/25.4, 0.0f, 480.0f,
		// 							0.0f, 3.6f*75/25, 270.0f,
		// 							0.0f, 0.0f, 1.0f);
		K.convertTo(K, CV_64F);

		// cv::FileStorage fs;
		// fs.open("camera_calibration.yml",cv::FileStorage::READ);
		// fs["camera_matrix"]>>K;

		// cout<<K;


		// for(int i=0;i<pnt1.size();i++) {
		// 	cout<<pnt1[i]<<" "<<pnt2[i]<<endl;
		// }

		Mat F = findFundamentalMat(pnt1, pnt2, CV_FM_RANSAC, 0.03, 0.99, noArray() );
		cout<<"got fund"<<endl<<F<<endl;
		cout<<type2str(K.type())<<" "<<type2str(F.type())<<endl;
		Mat E = K.t() * F * K;
		cout<<"got E"<<endl;

		SVD svd(E);
		Matx33d W(0,-1,0,   //HZ 9.13
		      1,0,0,
		      0,0,1);
		Matx33d Winv(0,1,0,
		     -1,0,0,
		     0,0,1);
		Mat R = svd.u * Mat(W) * svd.vt; //HZ 9.19
		Mat t = svd.u.col(2); //u3
		Matx34d P = Matx34d(1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0);
		Matx34d P1 = Matx34d(R.at<float>(0, 0),    R.at<float>(0,1), R.at<float>(0,2), t.at<float>(0),
		         R.at<float>(1,0),    R.at<float>(1,1), R.at<float>(1,2), t.at<float>(1),
		         R.at<float>(2,0),    R.at<float>(2,1), R.at<float>(2,2), t.at<float>(2));

		cout<<P1<<endl;

		vector<Point3d> pointcloud;
        vector<Point2f> correspImg1Pt;

		TriangulatePoints(pnt1, pnt2, K.inv(), P, P1, pointcloud, correspImg1Pt);
		for(int i=0;i<pointcloud.size();i++) {
			cout<<pointcloud[i].x<<" "<<pointcloud[i].y<<" "<<pointcloud[i].z<<" "<<endl;
		}
		break;
	}
	waitKey(0);

	return 0;
}