/*

//11/03/2023
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    // Read the two images from files
    cv::Mat img1 = cv::imread("/home/shen/MyFYP/image2.jpeg", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("/home/shen/MyFYP/image1.jpeg", cv::IMREAD_COLOR);
    cv::Size desiredsize(720,480);
    cv::resize(img1,img1,desiredsize);
    cv::resize(img2,img2,desiredsize);

    // Check if the images were loaded successfully
    if (img1.empty() || img2.empty()) {
        std::cout << "Error loading images\n";
        return -1;
    }

    cv::Mat gray_img1, gray_img2;

    // Convert the images to grayscale
    cv::cvtColor(img1, gray_img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray_img2, cv::COLOR_BGR2GRAY);

    // Create  SURF feature detectors
    // int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector1 = cv::xfeatures2d::SURF::create(2000);
    cv::Ptr<cv::xfeatures2d::SURF> detector2 = cv::xfeatures2d::SURF::create(2000);

    // Create a descriptor matcher object
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    // Get the start tick count
    int64 start = cv::getTickCount();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector1->detectAndCompute(gray_img1, cv::noArray(), keypoints1, descriptors1);
    detector2->detectAndCompute(gray_img2, cv::noArray(), keypoints2, descriptors2);

    // Match descriptors from the two images using Lowe's ratio test
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
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist < 3.0) {
            inliers.push_back(goodMatches[i]);
        }
    }
    // Visualize the matches on the grayscale images
    cv::Mat imageMatches;
    cv::drawMatches(gray_img1, keypoints1, gray_img2, keypoints2, inliers, imageMatches);
    cv::imshow("Matches", imageMatches);
    cv::imwrite("SURF_image_stitching_Matches.jpg", imageMatches);

    // Warp the second image onto the first image using the homography matrix
    cv::Mat warpedImg1;
    cv::warpPerspective(img1, warpedImg1, homography, cv::Size(img2.cols * 2, img2.rows));

    // Combine the two images into a single panoramic view
    cv::Mat half(warpedImg1, cv::Rect(0, 0, img1.cols, img1.rows));
    img2.copyTo(half);
    // Get the end tick count
    int64 end = cv::getTickCount();
    // Display the panoramic view
    cv::imshow("Panoramic View", warpedImg1);
    cv::imwrite("SURF_panoramic_view.jpg", warpedImg1);

// Output the number of detected feature points
    std::cout << "Number of feature points in image 1: " << keypoints1.size() << std::endl;
    std::cout << "Number of feature points in image 2: " << keypoints2.size() << std::endl;
    // Calculate the stitching time in seconds
    double stitching_time = (end - start) / cv::getTickFrequency();
    std::cout << "Stitching time: " << stitching_time << " seconds" << std::endl;
    cv::waitKey(0);

    return 0;
}
