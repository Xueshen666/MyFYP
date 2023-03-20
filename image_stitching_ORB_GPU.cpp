
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <iostream>

int main(int argc, char* argv[]){

  cv::Mat frame1 = cv::imread("/home/shen/MyFYP/image2.jpeg", cv::IMREAD_COLOR);
  cv::Mat frame2 = cv::imread("/home/shen/MyFYP/image1.jpeg", cv::IMREAD_COLOR);
  cv::Size desiredsize(720,480);
  cv::resize(frame1,frame1,desiredsize);
  cv::resize(frame2,frame2,desiredsize);
  
  if (frame1.empty() || frame2.empty()) {
    std::cout << "Error opening image files\n";
    return -1;
  }

  cv::Mat gray_frame1, gray_frame2, frameMatches, warpedFrame1;
  char key = 0;

  // Create CUDA ORB feature detectors
  int minHessian = 400;
  cv::Ptr<cv::cuda::ORB> detector1 = cv::cuda::ORB::create(minHessian);
  cv::Ptr<cv::cuda::ORB> detector2 = cv::cuda::ORB::create(minHessian);

  // Create CUDA device mats
  cv::cuda::GpuMat d_frame1, d_frame2;
  cv::cuda::GpuMat d_keypoints1, d_keypoints2;
  cv::cuda::GpuMat d_descriptors1, d_descriptors2;

  // Create a CUDA descriptor matcher
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

  // Convert the frames to grayscale
  cv::cvtColor(frame1, gray_frame1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame2, gray_frame2, cv::COLOR_BGR2GRAY);

  // Upload frames to the GPU
  d_frame1.upload(gray_frame1);
  d_frame2.upload(gray_frame2);
    // Get the start tick count
  int64 start = cv::getTickCount();
  // Detect ORB keypoints and compute descriptors on the GPU
  detector1->detectAndComputeAsync(d_frame1, cv::noArray(), d_keypoints1, d_descriptors1);
  detector2->detectAndComputeAsync(d_frame2, cv::noArray(), d_keypoints2, d_descriptors2);

  // Download keypoints and descriptors from the GPU
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  detector1->convert(d_keypoints1, keypoints1);
  detector2->convert(d_keypoints2, keypoints2);
  d_descriptors1.download(descriptors1);
  d_descriptors2.download(descriptors2);

  // Match descriptors from the two cameras using Lowe's ratio test
  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher->knnMatch(d_descriptors1, d_descriptors2, knnMatches, 2);

  std::vector<cv::DMatch> goodMatches;
  for (size_t i = 0; i < knnMatches.size(); i++) {
    if (knnMatches[i][0].distance < 0.7 * knnMatches[i][1].distance) {
      goodMatches.push_back(knnMatches[i][0]);
    }
  }

  // Compute homography matrix using RANSAC
  std::vector<cv::Point2f> pts1, pts2;
  for (size_t i = 0; i < goodMatches.size(); i++) {
    pts1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
    pts2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
  }
  cv::Mat homography = cv::findHomography(pts1, pts2, cv::RANSAC);
  // Remove outliers using the homography matrix
  std::vector<cv::DMatch> inliers;
  for (size_t i = 0; i < goodMatches.size(); i++) {
    cv::Mat pt1 = cv::Mat::ones(3, 1, CV_64F);
    pt1.at<double>(0, 0) = keypoints1[goodMatches[i].queryIdx].pt.x;
    pt1.at<double>(1, 0) = keypoints1[goodMatches[i].queryIdx].pt.y;

    cv::Mat pt2 = cv::Mat::ones(3, 1, CV_64F);
    pt2.at<double>(0, 0) = keypoints2[goodMatches[i].trainIdx].pt.x;
    pt2.at<double>(1, 0) = keypoints2[goodMatches[i].trainIdx].pt.y;

    cv::Mat projectedPt = homography * pt1;
    projectedPt /= projectedPt.at<double>(2, 0);

    double dx = projectedPt.at<double>(0, 0) - pt2.at<double>(0, 0);
    double dy = projectedPt.at<double>(1, 0) - pt2.at<double>(1, 0);
    double dist = std::sqrt(dx*dx + dy*dy);

    if (dist < 3.0) {
      inliers.push_back(goodMatches[i]);
    }
  }

    
  // Visualize the matches on the grayscale frames
  cv::drawMatches(gray_frame1, keypoints1, gray_frame2, keypoints2, inliers, frameMatches);
  cv::imshow("Matches", frameMatches);

  // Warp the second frame onto the first frame using the homography matrix
  cv::cuda::GpuMat d_warpedFrame1, gpuframe1;
  gpuframe1.upload(frame1);
  cv::cuda::warpPerspective(gpuframe1, d_warpedFrame1, homography, cv::Size(frame2.cols*2, frame2.rows));
  if (d_warpedFrame1.empty()) {
    std::cerr << "Failed to warp frame1" << std::endl;
    return -1;
  }

  d_warpedFrame1.download(warpedFrame1);
  gpuframe1.download(frame1);
  cv::Mat half(warpedFrame1, cv::Rect(0, 0, frame1.cols, frame1.rows));

  // Combine the two frames into a single panoramic view
  frame2.copyTo(half);
  // Get the end tick count
  int64 end = cv::getTickCount();
  cv::imshow("Panoramic View", warpedFrame1);

  cv::imwrite("panoramic_view_ORB_GPU.jpg", warpedFrame1);
  cv::imwrite("ORB_GPU_stitching_Matches.jpg", frameMatches);
  // Calculate the stitching time in seconds
  double stitching_time = (end - start) / cv::getTickFrequency();
  std::cout << "Stitching time: " << stitching_time << " seconds" << std::endl;

  cv::waitKey(0);

  return 0;
}
