#include "ImageTrans.h"
#include "cv.h"
#include <opencv2/opencv.hpp>

ImageTrans::ImageTrans()
{
}

vector<Mat> ImageTrans::getImageColorSplit(Mat& srcMat, int mode/*颜色分离颜色空间*/)
{
	vector<Mat> splitImg;
	if (srcMat.empty())
		return splitImg;

	Mat dstMat;
	switch (mode)
	{
	case IMG_COLORSPACE_HSV:
	{
		cv::cvtColor(srcMat, dstMat, CV_BGR2HSV);
	}
	case IMG_COLORSPACE_HSI:
	{
	}
	case IMG_COLORSPACE_BGR:
	{
	}
	case IMG_COLORSPACE_BGRA:
	{
		cv::cvtColor(srcMat, dstMat, CV_BGR2BGRA);
	}
	case IMG_COLORSPACE_HLS:
	{
		cv::cvtColor(srcMat, dstMat, CV_BGR2HSV);
	}
	default:
		break;
	}

	cv::split(dstMat, splitImg);
	return splitImg;
}

Mat ImageTrans::getTargetImage(Mat& srcmat, int minValue, int maxValue)
{
	if (srcmat.empty())
		return Mat();

	Mat dstMat = srcmat < maxValue;
	dstMat = dstMat >minValue;

	return dstMat;
}

ImageTrans::~ImageTrans()
{
}
