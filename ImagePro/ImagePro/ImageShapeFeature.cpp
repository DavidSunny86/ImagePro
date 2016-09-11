#include "ImageShapeFeature.h"


ImageShapeFeature::ImageShapeFeature()
{
}


float ImageShapeFeature::computerShapeContext(vector<Point>& src, vector<Point>& dst)
{
	cv::Ptr <cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();

	float dis = mysc->computeDistance(src, dst);

	return dis;
}

float ImageShapeFeature::computerShapeContext(Mat& src, Mat& dst, int n /*= 300*/, int width /*= 300*/, int height /*= 300*/)
{
	if (src.empty() || dst.empty())
	{
		return CTPLAR_INPUT_ERR;
	}

	Mat srcTmp;
	Mat dstTmp;
	Size sz2Sh(width, height);

	resize(src,srcTmp,sz2Sh);
	resize(dst, dstTmp, sz2Sh);

	vector<Point> srcShapeContext = shapeContextFeature(srcTmp);
	vector<Point> dstShapeContext = shapeContextFeature(dstTmp);

	cv::Ptr <cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();

	float dis = mysc->computeDistance(srcShapeContext, dstShapeContext);

	return dis;

}

/*************************************************
// Method: EllipticFourierDescriptor
// Description:ÕºœÒµƒÕ÷‘≤∏µ¡¢“∂√Ë ˆÀ„◊”
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

vector<Point> ImageShapeFeature::shapeContextFeature(const Mat& currentQuery, int n = 300)
{
	vector<vector<Point> > _contoursQuery;
	vector <Point> contoursQuery;
	findContours(currentQuery, _contoursQuery, RETR_LIST, CHAIN_APPROX_NONE);
	for (size_t border = 0; border < _contoursQuery.size(); border++)
	{
		for (size_t p = 0; p < _contoursQuery[border].size(); p++)
		{
			contoursQuery.push_back(_contoursQuery[border][p]);
		}
	}

	// In case actual number of points is less than n
	int dummy = 0;
	for (int add = (int)contoursQuery.size() - 1; add < n; add++)
	{
		contoursQuery.push_back(contoursQuery[dummy++]); //adding dummy values
	}

	// Uniformly sampling
	random_shuffle(contoursQuery.begin(), contoursQuery.end());
	vector<Point> cont;
	for (int i = 0; i < n; i++)
	{
		cont.push_back(contoursQuery[i]);
	}
	return cont;
}




ImageShapeFeature::~ImageShapeFeature()
{
}
