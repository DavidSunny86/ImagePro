#pragma once
#include "ImageProCom.h"


#define GRAY_THRESH 150
#define HOUGH_VOTE 100

class IMAGEPRO_MODEL_EXPORT ImageFFT
{
public:
	ImageFFT();
	Mat getFftImage(Mat& srcMat);
	Mat imageFftPre(Mat& srcMat);


private:
	Mat ImageFft_ocrCorrectingSample(Mat& srcMat);

	~ImageFFT();
};
