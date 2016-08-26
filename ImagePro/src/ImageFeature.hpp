#ifndef IMAGETEXTFEATURE_H_
#define IMAGETEXTFEATURE_H_
#endif

#include "ImageProCom.h"

#define GLCM_DIS 3               //灰度共生矩阵的统计距离  
#define GLCM_CLASS 16            //计算灰度共生矩阵的图像灰度值等级化  
#define GLCM_ANGLE_HORIZATION 0  //水平  
#define GLCM_ANGLE_VERTICAL   1  //垂直  
#define GLCM_ANGLE_DIGONAL    2  //对角  


typedef enum LBP_MODE
{
	LBP_GRAY_INVARIANCE = 0,
	LBP_ROTATION_INVARIANCE = 1,
	LBP_UNIFORM_INVARIANCE = 2,
	LBP_UNROTATION_INVARIANCE = 3,
	LBP_ROTATION_INVARIANCE_THR = 4,
	UNRESERVE
};


#define GRAY_BIN_CNT 64



class IMAGEPRO_MODEL_EXPORT ImageTextFeature
{
public:
	ImageTextFeature(Mat& srcMat);
	int calcGLCM(IplImage* bWavelet, int angleDirection, double* featureVector);

	void sheetLBPFeature(const Mat& src, const Mat& dst, int mode, int thr);
	int getHopCount(uchar i);
	void lbp59table(uchar* table);

	~ImageTextFeature();

private:
	Mat m_srcMat;
	
};

