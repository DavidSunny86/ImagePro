#pragma once

#include "ImageBaseUtils.h"
class ImageColor
{
public:
	ImageColor();

	//ÑÕÉ«ÌØÕ÷
	int computeColorFeature(const& Mat srcMat, cv::Mat& colorFeature);
	~ImageColor();
};

