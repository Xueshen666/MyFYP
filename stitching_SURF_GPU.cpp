
//
// Created by Zxs的MacBook on 2023/3/2.
//
//verison 10 : GOOD stitching using SUFT with FPS
// Created by Zxs的MacBook on 2023/3/2.
//
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::cuda;

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
    cv::Mat frameMatches;
    cv::Mat warpedFrame1;
    char key = 0;

    // Create CUDA SURF feature detectors
    int minHessian = 2000;
    cv::Ptr<cv::cuda::SURF_CUDA> detector1 = cv::cuda::SURF_CUDA::create(minHessian);
    cv::Ptr<cv::cuda::SURF_CUDA> detector2 = cv::cuda::SURF_CUDA::create(minHessian);

    // Create CUDA device mats
    cv::cuda::GpuMat d_frame1, d_frame2;
    cv::cuda::GpuMat d_keypoints1, d_keypoints2;
    cv::cuda::GpuMat d_descriptors1, d_descriptors2;

    // Create a CUDA descriptor matcher
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);

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

        // Upload frames to the GPU
        d_frame1.upload(gray_frame1);
        d_frame2.upload(gray_frame2);

        // Detect SURF keypoints and compute descriptors on the GPU
        detector1->detectWithDescriptors(d_frame1, cv::cuda::GpuMat(), d_keypoints1, d_descriptors1);
        detector2->detectWithDescriptors(d_frame2, cv::cuda::GpuMat(), d_keypoints2, d_descriptors2);
        //https://docs.opencv.org/4.x/db/d06/classcv_1_1cuda_1_1SURF__CUDA.html#ac6522a440dea4b95807d3a3b3417e6a0

        // Download keypoints and descriptors from the GPU
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        detector1->downloadKeypoints(d_keypoints1, keypoints1);
        detector2->downloadKeypoints(d_keypoints2, keypoints2);
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
        // Convert the keypoints to Point2f objects.
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

	//Warp the second frame onto the first frame using the homography matrix
	cv::cuda::GpuMat d_warpedFrame1,gpuframe1;
	gpuframe1.upload(frame1);
	cv::cuda::warpPerspective(gpuframe1,d_warpedFrame1, homography,cv::Size(frame2.cols*2,frame2.rows));
	if (d_warpedFrame1.empty()) {
	  std::cerr << "Failed to warp frame1" << std::endl;
	  return -1;
	}

	d_warpedFrame1.download(warpedFrame1);
	gpuframe1.download(frame1);
	cv::Mat half(warpedFrame1, cv::Rect(0,0,frame1.cols,frame1.rows));
	// Combine the two frames into a single panoramic view
	frame2.copyTo(half);
	// Display the panoramic view
	cv::imshow("Panoramic View",warpedFrame1);
        // Check if the user pressed 'q'
        key = cv::waitKey(1);
    }
    time(&end);

    //calculate the FPS
    double currentTotalTime = difftime(end,start);
    totalTime += currentTotalTime;
    std::cout << "TotalTime: " << totalTime <<"seconds"<<"\n"<<std::endl;
    double fps = frames / totalTime;
    std::cout << "FPS: " << fps <<"\n"<< std::endl;
    // Output the number of detected feature points
    std::cout << "Number of feature points in camera 1: " << d_keypoints1.size() << std::endl;
    std::cout << "Number of feature points in camera 2: " <<d_keypoints2.size() << std::endl;
    cv::imwrite("panoramic_view_SURF_GPU.jpg", warpedFrame1);
    cv::imwrite("SURf_GPU_stitching_Matches.jpg", frameMatches);
    // Release the VideoCapture objects and close all windows
    cap1.release();
    cap2.release();
    cv::destroyAllWindows();

    return 0;
}



