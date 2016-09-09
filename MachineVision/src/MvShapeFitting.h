#ifndef MVSHAPEFITTING_H_
#define MVSHAPEFITTING_H_

#include "imagebaseutils.h"
#include "mvImgThresold.h"


typedef enum MV_SHAPE_TYPE
{
    MV_CIRCLE,
    MV_LINE,
    MV_UNREV
};

class MvShapeFitting
{
public:
    MvShapeFitting();

    int circleFitting(std::vector<Point>& pointArr,float &fRadius, Point2f &center);
    int lineFitting(std::vector<Point>& pointArr,float& a, float &b);
    int rotateImg(Mat &src, Mat &dst, float angle, Point ptCenter /*= Point()*/, int clockWise /*= 1*/);


    ~MvShapeFitting() = default;
    MvShapeFitting(const MvShapeFitting& other) = default;
    MvShapeFitting(MvShapeFitting&& other) = default;
    MvShapeFitting& operator=(const MvShapeFitting& other) = default;
    MvShapeFitting& operator=(MvShapeFitting&& other) = default;
};


#endif
