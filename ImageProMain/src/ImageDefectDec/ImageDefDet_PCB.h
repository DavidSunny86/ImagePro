#ifndef IMAGEDEFDET_PCB_H_
#define IMAGEDEFDET_PCB_H_

#include "ImageDefDetUtils.h"
#include "ImageDefectDet.h"

class IMAGEDEFDET_MODEL_EXPORT ImageDefDet_PCB : ImageDefectDet
{
public:
	ImageDefDet_PCB();
	virtual Mat	image_Pre(Mat& srcMat, int materialType);
	virtual Mat image_Process(Mat& srcMat, Mat& mask);
	virtual ~ImageDefDet_PCB();
private:
	int m_materialType;
	Mat m_modelImage;
	
};

#endif