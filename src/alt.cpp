
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

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
#include <sstream>

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
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
  
  Eigen::Quaterniond q = rots[3];
  Eigen::Vector3d t = trans[3];
  Eigen::Affine3d aff = poses[3];
  Eigen::Matrix4d transform_mat = aff.matrix();
  Eigen::Matrix4d world_ref;
  std::cout<<transform_mat<<std::endl;

  std::cout << t << std::endl;
  
  
  // Converting Eigen type to Opencv type inputArray ---> Mat
  
  Eigen::Matrix3d mR = q.matrix();
  cv::Mat mr2 = cv::Mat::eye(3,3, CV_64F);
  eigen2cv(mR, mr2);
  cv::Mat T = cv::Mat::zeros(3,1,CV_64F);
  
  T.at<double>(0,0) =   t(0,0);
  T.at<double>(0,1) =   t(1,0);
  T.at<double>(0,2) =   t(2,0);

  int i=0;
  std::ostringstream oss;
  oss << "pair0" << i;
  std::cout << oss.str()<<std::endl;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile("pair04.pcd", *source);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_vx (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile("icp_conc_trans2.pcd", *target);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_vx (new pcl::PointCloud<pcl::PointXYZRGB>);
  
	pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>);
  
/***********************************************  
   Since Pc= R(Pw-t)
         Pw = R^-1(Pc)+t,
 ************************************************/
 
 //world_ref = (rotation.inverse())*transform_mat;
  
 //Voxel grid to downsample the Point cloud for faster computation

  pcl::VoxelGrid<pcl::PointXYZRGB> grid;
	grid.setLeafSize (12, 12, 12);
  grid.setInputCloud (source);
  grid.filter (*source_vx);
  
  grid.setInputCloud (target);
  grid.filter (*target_vx);

 
  //ICP for Point Cloud Alignment
  icp.setInputSource(source_vx);
  icp.setInputTarget(target_vx); 
  icp.align(*out);	
 
  //Transforming Point Cloud using Affine transformation
  pcl::transformPointCloud (*out, *aligned,transform_mat);

 //Concatenating two point clouds
  *aligned+=*source;	
  
	pcl::PCDWriter f;
	f.writeBinaryCompressed("aligned.pcd", *aligned);
	
	/* Please ignore this
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source0 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile("icp_conc_trans0.pcd", *source0);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile("icp_conc_trans1.pcd", *source1);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile("icp_conc_trans2.pcd", *source2);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source3 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile("icp_conc_trans3.pcd", *source3);
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source0vx (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source1vx (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source2vx (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source3vx (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr invx0 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr invx1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr invx2 (new pcl::PointCloud<pcl::PointXYZRGB>); 
   
     std::cout << "1"<<std::endl;
   
  grid.setInputCloud (source0);
  grid.filter (*source0vx);
  
  grid.setInputCloud (source1);
  grid.filter (*source1vx);

  grid.setInputCloud (source2);
  grid.filter (*source2vx);

  grid.setInputCloud (source3);
  grid.filter (*source3vx);


  *source1vx += *source0vx;
  
  grid.setInputCloud (source1vx);
  grid.filter (*invx0);
  
  *source2vx += *invx0 ;
  
  grid.setInputCloud (source2vx);
  grid.filter (*invx1);
  
  
  *source3vx += *invx1;
  
  pcl::PCDWriter fad;
	fad.writeBinaryCompressed("ebeb.pcd", *source3vx);
*/
  
	return 0;
}
