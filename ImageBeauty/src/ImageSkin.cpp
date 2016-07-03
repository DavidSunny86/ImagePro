#include "ImageSkin.h"


ImageSkin::ImageSkin()
{
}


cv::Mat ImageSkin::skinWhiteing(Mat& srcMat)
{
	if (srcMat.empty())
	{
		return Mat();
	}
	Mat outDst;
	Mat result;
	GaussianBlur(srcMat, outDst, Size(3,3),0,0);
	bilateralFilter(outDst, result, 5, 10, 2.5);
	Mat kernel(Size(3, 3), CV_32F,Scalar(0));
	kernel.at<float>(1, 1) = 5;
	kernel.at<float>(0, 1) = -1;
	kernel.at<float>(1, 2) = -1;
	kernel.at<float>(1, 0) = -1;
	kernel.at<float>(2, 1) = -1;
	cv::filter2D(result, result, result.depth(), kernel);
	return result;
}

cv::Mat skinWhiteing(Mat& srcMat,int bilateralSize,int sharpenSize)
{
	if (srcMat.empty())
	{
		return Mat();
	}
	Mat outDst;
	Mat result;
	GaussianBlur(srcMat, outDst, Size(3,3),0,0);
	bilateralFilter(outDst, result, bilateralSize, 2*bilateralSize, 0.5*bilateralSize);
	Mat kernel(Size(3, 3), CV_32F,Scalar(0));
	kernel.at<float>(1, 1) = sharpenSize;
	kernel.at<float>(0, 1) = -1;
	kernel.at<float>(1, 2) = -1;
	kernel.at<float>(1, 0) = -1;
	kernel.at<float>(2, 1) = -1;
	cv::filter2D(result, result, result.depth(), kernel);
	return result;
}

ImageSkin::~ImageSkin()
{
}
