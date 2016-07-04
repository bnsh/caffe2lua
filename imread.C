#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]) {
	cv::Mat img = cv::imread("/tmp/cat.jpg", -1);
	cv::Mat imgfloat;
	img.convertTo(imgfloat, CV_32FC3);

	float val1 = imgfloat.at<cv::Vec3d>(0,0)[0];
	fprintf(stderr, "%.7f\n", val1);

	return 0;
}
