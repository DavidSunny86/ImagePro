#include "MvShapeFitting.h"

MvShapeFitting::MvShapeFitting()
{}

int MvShapeFitting::circleFitting(std::vector<Point>& pointArr,float &fRadius, Point2f &center)
{
    if (points.size() < 3)
    {
        return CTPLAR_INPUT_ERR;
    }

	double X1 = 0, Y1 = 0;
	double X2 = 0, Y2 = 0;
	double X3 = 0, Y3 = 0;
	double X1Y1 = 0, X1Y2 = 0, X2Y1 = 0;
	double tempx, tempy;
	int nCount = points.size();
	for (int i = 0; i < nCount; i++)
	{
		tempx = points[i].x;
		tempy = points[i].y;
		X1 += tempx;
		Y1 += tempy;
		tempx = tempx * tempx;
		tempy = tempy * tempy;
		X2 += tempx;
		Y2 += tempy;

		X1Y1 += points[i].x * points[i].y;
		X2Y1 += points[i].y * tempx;
		X1Y2 += points[i].x * tempy;

		X3 += points[i].x * tempx;
		Y3 += points[i].y * tempy;
	}

	double C, D, E, G, H, N;
	N = nCount;
	C = N * X2 - X1 * X1;
	D = N * X1Y1 - X1 * Y1;
	E = N * X3 + N * X1Y2 - (X2 + Y2) * X1;
	G = N * Y2 - Y1 * Y1;
	H = N * X2Y1 + N * Y3 - (X2 + Y2) * Y1;

	double a = (H*D - E*G) / (C*G - D*D);
	double b = (H*C - E*D) / (D*D - G*C);
	double c = -(a*X1 + b*Y1 + X2 + Y2) / N;

	fRadius = sqrtf(a*a + b*b - 4.0 * c) / 2.0;
	center = Point2f(-a / 2.0, -b / 2.0);
	return CTPLAR_OK;
}

int MvShapeFitting::lineFitting(std::vector<Point>& pointArr,float& a, float &b)
{
    int nSize = points.size();
	if (nSize < 2)
    {
      return CTPLAR_INPUT_ERR;
    }

	double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	for (int i = 0; i < points.size(); ++i)
	{
		t1 += points[i].x * points[i].x;
		t2 += points[i].x;
		t3 += points[i].x * points[i].y;
		t4 += points[i].y;
	}
	b = (t3 * nSize - t2 * t4) / (t1 * nSize - t2 * t2);
	a = (t1 * t4 - t2 * t3) / (t1 * nSize - t2 * t2);
	return CTPLAR_OK;
}

int MvShapeFitting::rotateImg(Mat &src, Mat &dst, float angle, Point ptCenter /*= Point()*/, int clockWise /*= 1*/)
{
    if (clockWise != 1)
    {
        angle = -angle;
    }
    if (ptCenter.x == 0 && ptCenter.x == ptCenter.y)
    {
        ptCenter = Point(src.cols/2, src.rows/2);
    }
    Mat matRotate = getRotationMatrix2D(ptCenter, angle, 1.0);
    cv::warpAffine(src, dst, matRotate, dst.size());

    return CTPLAR_OK;
}



MvShapeFitting::~MvShapeFitting()
{}
