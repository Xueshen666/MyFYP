
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/opencv.hpp>


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {

    // Read the two images from disk
    cv::Mat img1 = cv::imread("/home/shen/MyFYP/image2.jpeg", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("/home/shen/MyFYP/image1.jpeg", cv::IMREAD_COLOR);
    
    // Define the image size for stitching
    cv::Size img_size = cv::Size(640, 480);

    // Create a Stitcher object
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    // Create a vector of images for stitching
    std::vector<cv::Mat> images;
    images.push_back(img1);
    images.push_back(img2);

    // Set the parameters for the Stitcher object
    cv::Stitcher::Status status = stitcher->estimateTransform(images);

    if (status != cv::Stitcher::OK) {
        std::cout << "Error stitching images: " << status << "\n";
        return -1;
    }
    // Get the start tick count
    int64 start = cv::getTickCount();
    cv::Mat panorama;
    status = stitcher->composePanorama(panorama);

    if (status != cv::Stitcher::OK) {
        std::cout << "Error stitching images: " << status << "\n";
        return -1;
    }
    // Get the end tick count
    int64 end = cv::getTickCount();
    // Calculate the stitching time in seconds
    double stitching_time = (end - start) / cv::getTickFrequency();

    // Display the stitched image
    cv::imshow("Panoramic View", panorama);
    cv::imwrite("opencv_stitcher_panoramic_view.jpg", panorama);

    cv::waitKey();

    // Release all windows
    cv::destroyAllWindows();

    std::cout << "Stitching time: " << stitching_time << " seconds" << std::endl;

    return 0;
}

