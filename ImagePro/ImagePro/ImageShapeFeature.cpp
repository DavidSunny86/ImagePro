#include "ImageShapeFeature.h"


ImageShapeFeature::ImageShapeFeature()
{
}



/*************************************************
// Method: EllipticFourierDescriptor
// Description:Í¼ÏñµÄÍÖÔ²¸µÁ¢Ò¶ÃèÊöËã×Ó
// Author: david.sun
// Date: 2016/08/25
// Returns: void
// Parameter: contour
// Parameter: CE
*************************************************/
void ellipticFourierDescriptor(vector<Point> &contour, vector<float> &CE)
{
	vector<float> ax, bx, ay, by;
	int m = contour.size();
	int n = 20;
	float t = (2 * cv::PI) / m;
	for (int k = 0; k < n; k++)
	{
		ax.push_back(0.0);
		bx.push_back(0.0);
		ay.push_back(0.0);
		by.push_back(0.0);
		for (int i = 0; i < m; i++)
		{
			ax[k] = ax[k] + contour[i].x*cos((k + 1)*t*i);
			bx[k] = bx[k] + contour[i].x*sin((k + 1)*t*i);
			ay[k] = ay[k] + contour[i].y*cos((k + 1)*t*i);
			by[k] = by[k] + contour[i].y*sin((k + 1)*t*i);
		}
		ax[k] = ax[k] / m;
		bx[k] = bx[k] / m;
		ay[k] = ay[k] / m;
		by[k] = by[k] / m;
	}

	for (int k = 0; k < n; k++)
	{
		float value = (float)sqrt((ax[k] * ax[k] + ay[k] * ay[k]) / (ax[0] * ax[0] + ay[0] * ay[0])) + sqrt((bx[k] * bx[k] + by[k] * by[k]) / (bx[0] * bx[0] + by[0] * by[0]));
		CE.push_back(value);

	}

}




ImageShapeFeature::~ImageShapeFeature()
{
}
