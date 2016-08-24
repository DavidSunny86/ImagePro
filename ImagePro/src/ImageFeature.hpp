#ifndef IMAGEFEATURE_H_
#define IMAGEFEATURE_H_
#endif

#include "ImageProCom.h"

#define GLCM_DIS 3               //�Ҷȹ��������ͳ�ƾ���  
#define GLCM_CLASS 16            //����Ҷȹ��������ͼ��Ҷ�ֵ�ȼ���  
#define GLCM_ANGLE_HORIZATION 0  //ˮƽ  
#define GLCM_ANGLE_VERTICAL   1  //��ֱ  
#define GLCM_ANGLE_DIGONAL    2  //�Խ�  


class IMAGEPRO_MODEL_EXPORT ImageFeature
{
public:
	ImageFeature(Mat& srcMat);
	int calcGLCM(IplImage* bWavelet, int angleDirection, double* featureVector);

	~ImageFeature();

private:
	Mat m_srcMat;
	
};

