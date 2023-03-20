//
// Created by Zxs的MacBook on 2023/3/7.
//
/*
 ORB perfect with FPS with number of feature points
//04/03/2023
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

int main(int argc, char* argv[])
{

    // Create two VideoCapture objects
    cv::VideoCapture cap1(1);
    cv::VideoCapture cap2(2);

    // Set the video frame width and height
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Set the frame rate
    cap1.set(cv::CAP_PROP_FPS, 30);
    cap2.set(cv::CAP_PROP_FPS, 30);

    // Check if the VideoCapture objects were opened successfully
    if (!cap1.isOpened() || !cap2.isOpened()) {
        std::cout << "Error opening video capture device(s)\n";
        return -1;
    }

    cv::Mat frame1, frame2;
    cv::Mat gray_frame1, gray_frame2;
    cv::Mat warpedFrame1;
    cv::Mat frameMatches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;

    char key = 0;

    // Create ORB feature detectors
    int minHessian = 500;
    cv::Ptr<cv::ORB> detector1 = cv::ORB::create(minHessian);
    cv::Ptr<cv::ORB> detector2 = cv::ORB::create(minHessian);

    // Create a descriptor matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    //Define variables to keep track of the total processing time and number of frames
    time_t start, end;
    double totalTime = 0.0;
    int frames = 0;
    time(&start);

    while (key != 'q') {
        // Read frames from the VideoCapture objects
        cap1.read(frame1);
        cap2.read(frame2);
        // Convert the frames to grayscale
        cv::cvtColor(frame1, gray_frame1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, gray_frame2, cv::COLOR_BGR2GRAY);

        // Detect ORB keypoints and compute descriptors
        cv::Mat descriptors1, descriptors2;
        detector1->detectAndCompute(gray_frame1, cv::noArray(), keypoints1, descriptors1);
        detector2->detectAndCompute(gray_frame2, cv::noArray(), keypoints2, descriptors2);

        // Match descriptors from the two cameras using Lowe's ratio test
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

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

	//frame calculataion
	frames++;

	// Visualize the matches on the grayscale frames

	cv::drawMatches(gray_frame1, keypoints1, gray_frame2, keypoints2, inliers, frameMatches);
	cv::imshow("Matches", frameMatches);

	// Warp the second frame onto the first frame using the homography matrix

	cv::warpPerspective(frame1, warpedFrame1, homography, cv::Size(frame2.cols * 2, frame2.rows));
	if (warpedFrame1.empty()) {
	  std::cerr << "Failed to warp frame1" << std::endl;
	  return -1;
	}
	cv::Mat half(warpedFrame1, cv::Rect(0, 0, frame1.cols, frame1.rows));

	// Combine the two frames into a single panoramic view
	frame2.copyTo(half);

	// Display the panoramic view
	cv::imshow("Panoramic View", warpedFrame1);

	// Check if the user pressed 'q'
	key = cv::waitKey(27);
    }

    time(&end);

    //calculate the FPS
    double currentTotalTime = difftime(end, start);
    totalTime += currentTotalTime;
    std::cout << "TotalTime: " << totalTime << " seconds" << "\n" << std::endl;
    double fps = frames / totalTime;
    std::cout << "FPS: " << fps << "\n" << std::endl;
    cv::imwrite("panoramic_view_ORB_CPU.jpg", warpedFrame1);
    cv::imwrite("ORB_CPU_stitching_Matches.jpg", frameMatches);
     // Output the number of detected feature points
    std::cout << "Number of feature points in camera 1: " << keypoints1.size() << std::endl;
    std::cout << "Number of feature points in camera 2: " << keypoints2.size() << std::endl;

    // Release the VideoCapture objects and close all windows
    cap1.release();
    cap2.release();
    cv::destroyAllWindows();

    return 0;
}
