#include "ImageSkin.h"


ImageSkin::ImageSkin()
{
}


cv::Mat ImageSkin::skinWhiteing(Mat& srcMat)
{
	Mat outDst;
	GaussianBlur(srcMat, outDst, Size(3,3),0,0);

	bilateralFilter(outDst, outDst, 9, 18, 4.5);

	Mat kernel(Size(3, 3), CV_32F,Scalar(-1));
	kernel.at<float>(1, 1) = 5;
	Mat result;
	cv::filter2D(outDst, result, outDst.depth(), kernel);
	return result;
}

ImageSkin::~ImageSkin()
{
}
