/***********************************************************************
References:

< Census Transform >  Robust and insensitive to various brightness and contrast images is the primary advantage of the transform.

1. "Non-parametric Local Transforms for Computing Visual Correspondence" - Ramin Zabih and John Woodall
2. https://hangyinuml.wordpress.com/2012/09/08/census-transform-c-implementation/
 
<Laplacian of Gaussian Filter (LoG) > Laplacian filters are derivative filters used to find areas of rapid change (edges) in images. Since derivative filters are very sensitive to noise, it is common to smooth the image (e.g., using a Gaussian filter) before applying the Laplacian. This two-step process is call the Laplacian of Gaussian (LoG) operation. This
enhances the image features as well as removing the effect
of intensity variations among images due to difference of
camera gains, ambient light


1. " Development of a Video-Rate Stereo Machine " , Takeo Kanade, Hiroshi Kano, Shigeru Kimura
Atsushi Yoshida, Kazuo Oda


**************************************************************************/



#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cmath>        // std::abs

#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <opencv2/core/eigen.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace cv;
using namespace std;

/**
 * Load data for this assignment.
 * @param fname The JSON input filename.
 * @param left_fnames The output left images of the stereo pair.
 * @param right_fnames The output right images of the stereo pair.
 * @param poses The 6D poses of the camera when the images were taken.
 *
 * This will probably throw an exception if there's something wrong
 * with the json file.
 */
 
 //This function creates a PCL visualizer, sets the point cloud to view and returns a pointer
//This function creates a PCL visualizer, sets the point cloud to view and returns a pointer

 
void LoadMetadata(const std::string& fname,
                  std::vector<std::string>& left_fnames,
                  std::vector<std::string>& right_fnames,
                  std::vector<Eigen::Affine3d>& poses,
		  std::vector<Eigen::Quaterniond>& rots,
		  std::vector<Eigen::Vector3d>& trans) {
  namespace bpt = boost::property_tree;
  bpt::ptree pt;
  bpt::read_json(fname, pt);
  for (bpt::ptree::iterator itr=pt.begin();
       itr != pt.end(); ++itr) {
    bpt::ptree::value_type v(*itr);
    bpt::ptree entry(v.second); 
    std::string left_fname( entry.get<std::string>("left") );
    std::string right_fname( entry.get<std::string>("right") );
    left_fnames.push_back(left_fname);
    right_fnames.push_back(right_fname);
    Eigen::Vector3d t(entry.get<double>("pose.translation.x"),
                      entry.get<double>("pose.translation.y"),
                      entry.get<double>("pose.translation.z"));
    Eigen::Quaterniond q(entry.get<double>("pose.rotation.w"),
                         entry.get<double>("pose.rotation.x"),
                         entry.get<double>("pose.rotation.y"),
                         entry.get<double>("pose.rotation.z"));
    Eigen::Affine3d aff = Eigen::Translation3d(t) * q;
    poses.push_back(aff);
    rots.push_back(q);
    trans.push_back(t);
  }
}

/**
 * Load calibration data.
 * Note this is basically the ROS CameraInfo message.
 * See
 * http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 * http://wiki.ros.org/image_pipeline/CameraInfo
 * for reference.
 *
 * Note: you probably don't need all the parameters ;)
 */
void LoadCalibration(const std::string& fname,
                     int &width,
                     int &height,
                     cv::Mat& D,
                     cv::Mat& K,
                     cv::Mat& R,
                     cv::Mat& P) {
  namespace bpt = boost::property_tree;
  bpt::ptree pt;
  bpt::read_json(fname, pt);
  width = pt.get<int>("width");
  height = pt.get<int>("height");
  {
    bpt::ptree &spt(pt.get_child("D"));
    D.create(5, 1, CV_32FC1);
    int i=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++i) {
      D.at<float>(i,0) = itr->second.get<float>("");
    }
  }
  {
    bpt::ptree &spt(pt.get_child("K"));
    K.create(3, 3, CV_32FC1);
    int ix=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++ix) {
      int i=ix/3, j=ix%3;
      K.at<float>(i,j) = itr->second.get<float>("");
    }
  }
  {
    bpt::ptree &spt(pt.get_child("R"));
    R.create(3, 3, CV_32FC1);
    int ix=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++ix) {
      int i=ix/3, j=ix%3;
      R.at<float>(i,j) = itr->second.get<float>("");
    }
  }
  {
    bpt::ptree &spt(pt.get_child("P"));
    P.create(3, 4, CV_32FC1);
    int ix=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++ix) {
      int i=ix/4, j=ix%4;
      P.at<float>(i,j) = itr->second.get<float>("");
    }
  }
}


void preprocess_LoG( Mat& src, Mat&  lap_left)
	{
   
  Mat src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  int c;
  
  //removing noise using gaussian blur
  
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  cvtColor( src, src_gray, CV_RGB2GRAY );
  
  //Equalizing illumination
  equalizeHist(src_gray, src_gray);
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  /// Apply Laplace transform

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, lap_left );

  imshow(" LoG" , lap_left );

  waitKey(0);	
  
}


void preproces_sobel(Mat& src, Mat& grad)
{
  
  Mat src_gray;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert it to grayscale
  cvtColor( src, src_gray, CV_RGB2GRAY );
  
  //Equalizing illumination
  equalizeHist(src_gray, src_gray);
  /// Declare variables grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow("sobel gradient", grad );

  waitKey(0);

}

void censustransform(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp)
	{
	 int h = left.rows;
	 int w = left.cols;
   Mat temp(left.size(), CV_8U);
	 unsigned int census = 0;
   unsigned int bit = 0;
   int m=3;
   int n=3; //window size 3x3
   int i,j,x,y;
   int shiftCount = 0;
   for (x = m/2; x < h - m/2; x++)
		{
   		for(y = n/2; y < w - n/2; y++)
   			{
     			census = 0;
     			shiftCount = 0;
     			for (i = x - m/2; i <= x + m/2; i++) // centre pixel is compared with every other pixel
     				{
       				for (j = y - n/2; j <= y + n/2; j++)
       					{
 
         					if( shiftCount != m*n/2 ) //skip the center pixel
         						{
							         census <<= 1;  //left shift bit
         								if( left.at<uchar>(i,j) < left.at<uchar>(x,y) )//compare pixel values in the neighborhood
         									bit = 1;
         								else
								          bit = 0;
							         census = census + bit; 
         						}
         					shiftCount ++;
       					}	
     				}
          
			    temp.ptr<uchar>(x)[y] = census;
	   		}
 		}
 	 //temp.copyTo(disp);
 	 normalize(temp, disp, 0, 255, CV_MINMAX, CV_8U);
   imshow ("census", disp);
   waitKey(27);
	}
	 
int colorDiff(const Vec3b &p1, const Vec3b &p2)
{
    int colorDiff, diff = 0;

    for(uchar color = 0; color < 3; color++)
    {
        colorDiff = std::abs(p1[color] - p2[color]);
        diff = (diff > colorDiff)? diff: colorDiff;
    }

    return diff;
}

void post_process(const Mat& left, Mat& src) 
	{
    int	newVal=40;
    medianBlur(src, src, 3);
	  //filterSpeckles(src,newVal,1,1);
    //imshow ("speckle_filtered	", src);
        
    //code for interpolation
    /*
    const int DISP_OCCLUSION = 1;
		// Minimum and maximum disparity
		int dMin=0;
		int dMax=30;  
  
  	Size dispSize = src.size();
    Mat dispTemp(dispSize, CV_32S);
    int maxSearchDepth=20;
    
     // look on the 16 different directions
    int directionsW[] = {0, 2, 2, 2, 0, -2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1};
    int directionsH[] = {2, 2, 0, -2, -2, -2, 0, 2, 2, 1, -1, -2, -2, -1, 1, 2};

    // loop on the whole picture
    for(size_t h = 0; h < dispSize.height; h++)
    {
        for(size_t w = 0; w < dispSize.width; w++)
        {
            // if the pixel is not an outlier
            if(src.at<int>(h, w) >= dMin)
            {
                dispTemp.at<int>(h, w) = src.at<int>(h, w);
            }
            else
            {
                vector<int> neighborDisps(16, src.at<int>(h, w));
                vector<int> neighborDiffs(16, -1);
                for(uchar direction = 0; direction < 16; direction++)
                {
                    int hD = h, wD = w;
                    bool inside = true, gotDisp = false;
                for(uchar sD = 0; sD < maxSearchDepth && inside && !gotDisp; sD++)
                {
                    int hD = h, wD = w;
                    bool inside = true, gotDisp = false;
                    for(uchar sD = 0; sD < maxSearchDepth ; sD++)
                    {
                        // go one step further
                        if(sD % 2 == 0)
                        {
                            hD += directionsH[direction] / 2;
                            wD += directionsW[direction] / 2;
                        }
                        else
                        {
                            hD += directionsH[direction] - directionsH[direction] / 2;
                            wD += directionsW[direction] - directionsW[direction] / 2;
                        }
                        inside = hD >= 0 && hD < dispSize.height && wD >= 0 && wD < dispSize.width;
                        if(inside && src.at<int>(hD, wD) >= dMin)
                        {
                            neighborDisps[direction] = src.at<int>(hD, wD);
                            neighborDiffs[direction] = colorDiff(left.at<Vec3b>(h, w), left.at<Vec3b>(hD, wD));
                            gotDisp = true;
                        }
                    }

                }

                if(src.at<int>(h, w) == dMin - DISP_OCCLUSION)
                {
                    int minDisp = neighborDisps[0];
                    for(uchar direction = 1; direction < 16; direction++)
                    {
                        if(minDisp > neighborDisps[direction])
                            minDisp = neighborDisps[direction];
                    }
                    dispTemp.at<int>(h, w) = minDisp;
                }
                else
                {
                    int minDisp = neighborDisps[0];
                    int minDiff = neighborDiffs[0];
                    for(uchar dir = 1; dir < 16; dir++)
                    {
                        if(minDiff < 0 || (minDiff > neighborDiffs[dir] && neighborDiffs[dir] > 0))
                        {
                            minDisp = neighborDisps[dir];
                            minDiff = neighborDiffs[dir];
                        }
                    }
                    dispTemp.at<int>(h, w) = minDisp;
                }
            }
        }
    }

    dispTemp.copyTo(src);
	}
	*/
}
	
						

/*
void ComputeDisparity(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp)
 {

	Mat dispar, g1, g2;
	cvtColor(left, g1,CV_BGR2GRAY);
	cvtColor(right, g2, CV_BGR2GRAY);

	StereoSGBM sbm;
  sbm.SADWindowSize = 3;
  sbm.numberOfDisparities = 144;
  sbm.preFilterCap = 63;
  sbm.minDisparity = -39;
  sbm.uniquenessRatio = 10;
  sbm.speckleWindowSize = 100;
  sbm.speckleRange = 32;
  sbm.disp12MaxDiff = 1;
  sbm.fullDP = false;
  sbm.P1 = 216;
  sbm.P2 = 864;
  sbm(g1, g2, disp);
  sbm(g1, g2, dispar);
  normalize(dispar, disp, 0, 255, CV_MINMAX, CV_8U);
  imshow("left1",left);
  imshow("right1", right);
  imshow("disparity", disp); 
  std::cout<<"Press any key ...."<<std::endl;
  waitKey(0);
  
  Mat dispSGBMn, dispSGBMheat;
  normalize(disp, dispSGBMn, 0, 255, CV_MINMAX, CV_8U); // form 0-255
  equalizeHist(dispSGBMn, dispSGBMn);    
  //imshow( "WindowDispSGBM", dispSGBMn );

    applyColorMap(dispSGBMn, dispSGBMheat, COLORMAP_JET);
    imshow( "WindowDispSGBMheat", dispSGBMheat );
    waitKey(1);

}
*/

int main(int argc, char *argv[]) {

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " JSON_DATA_FILE JSON_LEFT_CALIB_FILE JSON_RIGHT_CALIB_FILE\n";
    return -1;
  }

  // load metadata 
  std::vector<std::string> left_fnames, right_fnames;
  std::vector<Eigen::Affine3d> poses;
  std::vector<Eigen::Vector3d> trans;
  std::vector<Eigen::Quaterniond> rots;
  LoadMetadata(argv[1], left_fnames, right_fnames, poses, rots, trans);


  // load calibration info.
  // note: you should load right as well
  int left_w, left_h;
  cv::Mat left_D, left_K, left_R, left_P;
  LoadCalibration(argv[2], left_w, left_h, left_D, left_K, left_R, left_P);
   
  int right_w, right_h;
  cv::Mat right_D, right_K, right_R, right_P;
  LoadCalibration(argv[3], right_w, right_h, right_D, right_K, right_R, right_P);
  
  


  // here you should load the images from their filenames
  // NOTE: make sure you run the program from the data/ directory
  // for the paths to work.
  // alternatively feel free to modify the input json file or the image
  // filenames at runtime so the images are found.

  // load one of the images as an example.
  
  std::cout << "loading " << left_fnames[0] << " ... ";
  cv::Mat left = cv::imread(left_fnames[0]);
  if (left.empty()) {
    std::cerr << "image not found.\n";
    return -1;
  } else {
    std::cout << "loaded image file with size " << left.cols << "x" << left.rows << "\n";
  }


  cv::Mat right = cv::imread(right_fnames[0]);


  std::cout << "string test 1" << std::endl;
 
  double left_fovx;
  double left_fovy;
  double right_fovx;
  double right_fovy;
  double left_focalLength;
  double right_focalLength;
  cv::Point2d left_principalPoint;
  cv::Point2d right_principalPoint;
  double left_aspectRatio;
  double right_aspectRatio;
  
  calibrationMatrixValues(left_K, left.size(), left_w, left_h, left_fovx, left_fovy, left_focalLength, left_principalPoint, left_aspectRatio);
  
  calibrationMatrixValues(right_K, right.size(), right_w, right_h, right_fovx, right_fovy, right_focalLength, right_principalPoint, right_aspectRatio);

  std::cout<<left_focalLength<<std::endl;
  std::cout<<left_fovy<<std::endl;
  std::cout<<left_principalPoint<<std::endl;
  std::cout<<right_focalLength<<std::endl;
  std::cout<<right_fovy<<std::endl;
  std::cout<<right_principalPoint<<std::endl;
  
  Mat R1, R2, Q;
  Eigen::Quaterniond q = rots[0];
  Eigen::Vector3d t = trans[0];
  std::cout << q.matrix() << std::endl;
  std::cout << t << std::endl;
  
  // Converting Eigen type to Opencv type inputArray ---> Mat
  
  Eigen::Matrix3d mR = q.matrix();
  cv::Mat mr2 = cv::Mat::eye(3,3, CV_64F);
  eigen2cv(mR, mr2);
  cv::Mat T = cv::Mat::zeros(3,1,CV_64F);
  
  T.at<double>(0,0) =   t(0,0);
  T.at<double>(0,1) =   t(1,0);
  T.at<double>(0,2) =   t(2,0);

  std::cout << t << std::endl;
  
  //scaling down the disparity image by 1/16 given by the opencv documentation http://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  
  
  //Computes rectification transforms for each head of a calibrated stereo camera.
  //with this I get the Q matrix for reprojectImageTo3D()

  stereoRectify(left_K,left_D,right_K,right_D,left.size(),mr2,T,R1,R2,left_P,right_P,Q);

  Mat disp;
  Mat left_copy,right_copy;
  left.copyTo(left_copy);
  left.copyTo(right_copy);
	
	Mat pre_left,pre_right;
	
	//preproces_sobel(left,pre_left);
	preprocess_LoG(left,pre_left);
	preprocess_LoG(right,pre_right);  

  //ComputeDisparity(left, right, disp);
  
  censustransform(pre_left, right, disp);
  
  imshow("disparity map from census transform" , disp);
  waitKey(0);
  
 // post_process(left,disp);//), disp);

  Mat recons3D(disp.size(), CV_32FC3);

  bool handleMissingValues=true; 
 
 /* 
   double baseline, cx, cy, f, cx_s;
        baseline= 0.18;//std::abs(right_principalPoint - left_principalPoint);
        cx = disp.cols/2; // set to midpoint of the image 
        cy = disp.rows/2; // set to midpoint of the image 
        f = 827.422; // calculate the focal length in Px
        cx_s =disp.cols/2; //set to midpoint of the image 

        double pot[4][4] = { {1, 0,   0,   -cx }, // write to q mat
                           {0, 1,   0,   -cy },
                           {0, 0,   0,    f  },
                           {0, 0, (-1/baseline), ((cx-cx_s)/baseline) }};
         Mat Q( 4, 4, CV_64FC1,pot);
 */

  reprojectImageTo3D( disp, recons3D, Q, handleMissingValues, CV_32F);
    
  std::cout<<Q<<std::endl;
  imshow("ShowRecons3D",recons3D);
  
  std::cout<<"Now  press escape key" << std::endl;
  waitKey(0);
  
  // finally compute the output point cloud from one or more stereo pairs.
  //
  // This is just a silly example of creating a colorized XYZ RGB point cloud.
  // open it with pcl_viewer. then press 'r' and '5' to see the rgb.

  std::cout << "Creating Point Cloud ..." <<std::endl;

  std::cout << left.size()<<std::endl;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>); 

    for (int rows = 0; rows < recons3D.rows; ++rows)
     { 
        for (int cols = 0; cols < recons3D.cols; ++cols) 
        { 
            uchar* disp_ptr = disp.ptr<uchar>(rows);
            cv::Point3f point = recons3D.at<cv::Point3f>(rows, cols); 
            
            // removing points of zero disparity 
            uchar d = disp_ptr[cols];
            if ( d == 0 ) 
            	continue;
  
            pcl::PointXYZ pcl_point(point.x, point.y, point.z); // normal PointCloud 
            pcl::PointXYZRGB pcl_point_rgb;
            pcl_point_rgb.x = point.x;    // rgb PointCloud 
            pcl_point_rgb.y = point.y; 
            pcl_point_rgb.z = point.z; 
            // image_left is the stereo rectified image used in stere reconstruction 
           cv::Vec3b intensity = left.at<cv::Vec3b>(rows,cols); //BGR 
           uint32_t rgb = (static_cast<uint32_t>(intensity[2]) << 16 | static_cast<uint32_t>(intensity[1]) << 8 | static_cast<uint32_t>(intensity[0])); 
            pcl_point_rgb.rgb = *reinterpret_cast<float*>(&rgb);

            cloud_xyz->push_back(pcl_point); 
            cloud_xyzrgb->push_back(pcl_point_rgb); 
           } 
        } 
    
  
  
  
  std::cout << "saving a pointcloud to out.pcd\n";
  //pcl::io::savePCDFileASCII("out.pcd", pc);
  pcl::PCDWriter w;
  w.writeBinaryCompressed("users.pcd", *cloud_xyz);
  pcl::PCDWriter aw;
  aw.writeBinaryCompressed("extra.pcd", *cloud_xyzrgb);
  //pcl::io::savePCDBinaryCompressed("out.pcd", pc);
    
  return 0;
  
}
