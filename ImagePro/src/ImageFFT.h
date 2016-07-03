#pragma once
#include "ImageProCom.h"

class IMAGEPRO_MODEL_EXPORT ImageFFT
{
public:
	ImageFFT();
	Mat getFftImage(Mat& srcMat);
	~ImageFFT();
};
