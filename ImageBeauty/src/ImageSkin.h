#pragma once
#include "ImageBeauityUtils.h"

class IMAGEBEAUITY_MODEL_EXPORT ImageSkin
{
public:
	ImageSkin();
	Mat skinWhiteing(Mat& srcMat);
	
	~ImageSkin();
};

