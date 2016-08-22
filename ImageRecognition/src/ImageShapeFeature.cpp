#include "ImageBaseRec.h"
#include "ImageShapeFeature.h"


Mat imageSpFtPrePro(Mat& src, Size normSize)
{
	Mat resizeDst;
	return resize(src,resizeDst,normSize);
}

vector<Point> getImageFtVector(const Mat& src, int thr)
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

int getShapeFtDis(vector<Point>, vector<Point>, float& shapeDis)
{
	cv::Ptr <cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();
	shapeDis = mysc->computeDistance(contQuery, contii);
	return 0;
}

