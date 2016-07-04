#pragma once
#include "ImageDefDetUtils.h"


class IMAGEDEFDET_MODEL_EXPORT ImageDefectDet
{
public:
	ImageDefectDet();
	virtual Mat	image_Pre(Mat& srcMat, int materialType);
	virtual Mat image_Process(Mat& srcMat);
	virtual Mat image_Process(Mat& srcMat, Mat& mask);
	virtual Mat setModelImage(Mat& modelImg){
		if (modelImg.empty())return Mat();
		m_modelImage = modelImg;
	}
	virtual Mat setModelImage(string fileName);
	virtual ~ImageDefectDet();

private:
	int m_materialType;
	Mat m_modelImage;

};

