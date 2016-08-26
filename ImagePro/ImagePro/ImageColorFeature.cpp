#include "ImageColorFeature.h"


ImageColorFeature::ImageColorFeature()
{
}


ImageColorFeature::~ImageColorFeature()
{
}



int ImageColorFeature::computeColorFeature(const& Mat srcMat, cv::Mat& colorFeature)
{
	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float hranges[2] = { 0, 255 };
	const float* ranges[1] = { hranges };
	calcHist(&srcMat, 1, channels, Mat(), colorFeature, 1, histSize, ranges);
	return CTPLAR_OK;
}