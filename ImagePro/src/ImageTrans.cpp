#include "ImageTrans.h"
#include "cv.h"
#include <opencv2/opencv.hpp>

ImageTrans::ImageTrans()
{
}



/*************************************************
// Method: getImageColorSplit
// Description:将图片分离成不同的颜色空间
// Author: david.sun
// Date: 2016/08/22
// Returns: vector<Mat>
// Parameter: srcMat
// Parameter: mode
*************************************************/
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
	break;
	case IMG_COLORSAPCE_LAB:
	{
		cv::cvtColor(srcMat,dstMat, CV_BGR2Lab);
	}
	break;
	case IMG_COLORSPACE_BGR:
	{
	}
	break;
	case IMG_COLORSPACE_BGRA:
	{
		cv::cvtColor(srcMat, dstMat, CV_BGR2BGRA);
	}
	break;
	case IMG_COLORSPACE_HLS:
	{
		cv::cvtColor(srcMat, dstMat, CV_BGR2HSV);
	}
	break;
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
