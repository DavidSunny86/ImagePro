#pragma once

#include "ImageBaseUtils.h"
class ImageColor
{
public:
	ImageColor();

	//��ɫ����
	int computeColorFeature(const& Mat srcMat, cv::Mat& colorFeature);
	~ImageColor();
};

