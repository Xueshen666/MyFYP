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

int main(int argc, char* argv[]) {

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

    // Define the image size for stitching
    cv::Size img_size = cv::Size(640, 480);

    // Create a vector of images for stitching
    std::vector<cv::Mat> images;

    char key = 0;
    //Define variables to keep track of the total processing time and number of frames
    time_t start, end;
    double totalTime = 0.0;
    int frames = 0;
    time(&start);
    while (key != 'q') {
      // Read frames from the VideoCapture objects
      cap1.read(frame1);
      cap2.read(frame2);

      // Add the frames to the vector of images
      images.push_back(frame1);
      images.push_back(frame2);

      // Check if the vector has at least two images
      if (images.size() >= 2) {
	// Create a Stitcher object
	cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

	// Set the parameters for the Stitcher object
	cv::Stitcher::Status status = stitcher->estimateTransform(images);

	if (status != cv::Stitcher::OK) {
	  std::cout << "Error stitching images: " << status << "\n";
	}
	else {
	  cv::Mat panorama;
	  status = stitcher->composePanorama(panorama);

	  if (status != cv::Stitcher::OK) {
	    std::cout << "Error stitching images: " << status << "\n";
	  }
	  else {
	    // Display the stitched image
	    cv::imshow("Panoramic View", panorama);
	    cv::imwrite("Opencv_stitcher_module_video_panoramic_view.jpg", panorama);
	  }
	}
	frames ++;
	// Clear the vector of images for the next iteration
	images.clear();
      }

      // Check if the user pressed 'q'
      key = cv::waitKey(27);
    }
    time(&end);

    //calculate the FPS
    double currentTotalTime = difftime(end,start);
    totalTime += currentTotalTime;
    std::cout << "TotalTime: " << totalTime <<"seconds"<<"\n"<<std::endl;
    double fps = frames / totalTime;
    std::cout << "FPS: " << fps <<"\n"<< std::endl;
    // Release the VideoCapture objects and close all windows
    cap1.release();
    cap2.release();
    cv::destroyAllWindows();

    return 0;
}
