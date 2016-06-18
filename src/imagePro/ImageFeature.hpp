#ifndef IMAGEFEATURE_H_
#define IMAGEFEATURE_H_
#endif

#include "ImageProCom.h"


class IMAGEPRO_MODEL_EXPORT ImageFeature
{
public:
	ImageFeature(Mat& srcMat);
	~ImageFeature();

private:
	Mat m_srcMat;
	
};

