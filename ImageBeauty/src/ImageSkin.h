#pragma once
#include "ImageBeautyUtils.h"

class IMAGEBEAUITY_MODEL_EXPORT ImageSkin
{
public:
	ImageSkin();
	Mat skinWhiteing(Mat& srcMat);
	Mat skinWhiteing(Mat& srcMat,int bilateralSize,int sharpenSize);
	~ImageSkin();
};

