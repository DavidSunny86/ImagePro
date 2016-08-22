#ifndef IMAGESHAPEFEATURE_H_
#define IMAGESHAPEFEATURE_H_

#include "ImageBaseRec.h"


extern Mat imageSpFtPrePro(Mat& src,Size normSize);

extern vector<Point> getImageFtVector(const Mat& src, int thr);

extern int    getShapeFtDis(vector<Point>, vector<Point>, float* shapeDis);



#endif