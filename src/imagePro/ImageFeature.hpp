#ifndef IMAGEFEATURE_H_
#define IMAGEFEATURE_H_
#endif

#include "ImageProCom.h"
#include "cv.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace  cv;

class IMAGEPRO_MODEL_EXPORT ImageFeature
{
public:
	ImageFeature(Mat& srcMat);
	~ImageFeature();

private:
	Mat m_srcMat;
	
};

