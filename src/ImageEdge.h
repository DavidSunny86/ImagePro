#ifndef IMAGEEDGE_H_
#define IMAGEEDGE_H_

#include "ImageProCom.h"

class IMAGEPRO_MODEL_EXPORT ImageEdge
{
	typedef enum  IMAGEEDGE_MODEL
	{
		IMAGEEDGE_BLUR,
		IMAGEEDGE_SOBEL,
		IMAGEEDGE_CANNY,
		IMAGEEDGE_LAPLACIAN,
		IMAGEEDGE_LOG,
		IMAGEEDGE_PRIW,
		IMAGEEDGE_UNRES
	};

public:
	ImageEdge(Mat& srcMat);
	Mat getImageResult(Mat& srcMat,int iModel);
	~ImageEdge();

private:
	Mat m_srcMat;
	int m_iResult;


};


#endif // !imageedge_h_



