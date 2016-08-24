#ifndef IMAGEFEATURE_H_
#define IMAGEFEATURE_H_
#endif

#include "ImageProCom.h"

#define GLCM_DIS 3               //灰度共生矩阵的统计距离  
#define GLCM_CLASS 16            //计算灰度共生矩阵的图像灰度值等级化  
#define GLCM_ANGLE_HORIZATION 0  //水平  
#define GLCM_ANGLE_VERTICAL   1  //垂直  
#define GLCM_ANGLE_DIGONAL    2  //对角  


class IMAGEPRO_MODEL_EXPORT ImageFeature
{
public:
	ImageFeature(Mat& srcMat);
	int calcGLCM(IplImage* bWavelet, int angleDirection, double* featureVector);

	~ImageFeature();

private:
	Mat m_srcMat;
	
};

