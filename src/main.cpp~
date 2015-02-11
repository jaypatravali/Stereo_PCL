
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cmath>        
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

#include <pcl/io/ply_io.h>


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
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


//Stero SGBM was used compute disparity map.
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
  waitKey(0);
  
  Mat  dispSGBMheat,dispSGBMn;
  normalize(disp, dispSGBMn, 0, 255, CV_MINMAX, CV_8U); // from 0-255
  equalizeHist(dispSGBMn, dispSGBMn);   
  //Red closer points and Blue deeper. Color Gradient to show depth
  applyColorMap(dispSGBMn, dispSGBMheat, COLORMAP_JET);
  imshow( "WindowDispSGBMheat", dispSGBMheat );
  waitKey(1);

}

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
  
  std::cout << "loading " << left_fnames[2] << " ... ";
  cv::Mat left = cv::imread(left_fnames[2]);
  if (left.empty()) {
    std::cerr << "image not found.\n";
    return -1;
  } else {
    std::cout << "loaded image file with size " << left.cols << "x" << left.rows << "\n";
  }

  std::cout << "loading " << right_fnames[2] << " ... ";
  cv::Mat right = cv::imread(right_fnames[2]);
 
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
 

  /*
  Extraction of :
  1. rotation matrix 3x3 
  2. Translation vector (3,1)
  3. Affine transformation Matrix 4x4 */
  
  Eigen::Quaterniond q = rots[2];
  Eigen::Vector3d t = trans[2];
  std::cout << q.matrix() << std::endl;
  std::cout << t << std::endl;
  Eigen::Affine3d aff = poses[2];
  Eigen::Matrix4d transform_mat = aff.matrix();
  
  // Converting Eigen type to Opencv type inputArray ---> Mat type of opencv
  
  Mat R1, R2, Q;
  Eigen::Matrix3d mR = q.matrix();
  cv::Mat mr2 = cv::Mat::eye(3,3, CV_64F);
  eigen2cv(mR, mr2);
  cv::Mat T = cv::Mat::zeros(3,1,CV_64F);
  
  T.at<double>(0,0) =   t(0,0);
  T.at<double>(0,1) =   t(1,0);
  T.at<double>(0,2) =   t(2,0);


  //scaling down the disparity image by 1/16 given by the opencv documentation http://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  
  
  //Computes rectification transforms for each head of a calibrated stereo camera.
  //with this I get the Q matrix for reprojectImageTo3D()

  stereoRectify(left_K,left_D,right_K,right_D,left.size(),mr2,T,R1,R2,left_P,right_P,Q);
  
  cv::Mat disp;
  ComputeDisparity(left, right, disp);
  
  

  Mat dispSGBMscale; 
  //disp.convertTo(dispSGBMscale,CV_32F, 1./16); 
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
  std::cout<<"Now  press escape key5" << std::endl;
  Mat recons3D(disp.size(), CV_32FC3);

  bool handleMissingValues=true; 
  std::cout<<"Now  press escape key4" << std::endl;
  reprojectImageTo3D( disp, recons3D, Q, handleMissingValues, CV_32F);

  imshow("ShowRecons3D",recons3D);
  
  std::cout<<"Now  press escape key" << std::endl;
  waitKey(27);
  
  
   
  // finally compute the output point cloud from one or more stereo pairs.
  //
  // This is just a silly example of creating a colorized XYZ RGB point cloud.
  // open it with pcl_viewer. then press 'r' and '5' to see the rgb.

  std::cout << "Creating Point Cloud ..." <<std::endl;
  //std::cout <<dispSGBMscale.size() <<std::endl;
  std::cout << left.size()<<std::endl;
  

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>); 
       

    for (int rows = 0; rows < recons3D.rows; ++rows) 
    { 
        
        for (int cols = 0; cols < recons3D.cols; ++cols) 
          { 
               
         	 cv::Point3f point = recons3D.at<cv::Point3f>(rows, cols);
                 		
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
  
  //colorized point cloud
  pcl::PCDWriter w;
  w.writeBinaryCompressed("users.pcd", *cloud_xyz);
  pcl::PCDWriter aw;
  aw.writeBinaryCompressed("pair003.pcd", *cloud_xyzrgb);
  //pcl::io::savePCDBinaryCompressed("out.pcd", pc);

  return 0;
  
}
