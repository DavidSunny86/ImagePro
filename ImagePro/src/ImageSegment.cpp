#include "ImageSegment.h"

ImageSegment::ImageSegment()
{}

bool ImageSegment::imageSegment(Mat& src, Mat& dst, int segmentMode, int spatialRad,int colorRad, int maxPryLevel)
{
	if (src.empty())
	{
		return false;
	}

	switch (segmentMode)
	{
	case 0:
	{
		pyrMeanShiftFiltering(src, dst, spatialRad, colorRad, maxPryLevel);
		RNG rng = theRNG();
		Mat mask(dst.rows + 2, dst.cols + 2, CV_8UC1, Scalar::all(0));
		for (int i = 0; i < dst.rows; i++)
		{
			for (int j = 0; j < dst.cols; j++)
			{
				if (mask.at<uchar>(i + 1, j + 1) == 0)
				{
					Scalar newcolor(rng(256), rng(256), rng(256));
					floodFill(dst, mask, Point(j, i), newcolor, 0, Scalar::all(1), Scalar::all(1));
				}
			}
		}
	}
	break;
	default:
		break;
	}

	return  true;
}


ImageSegment::~ImageSegment()
{}



int main(int argc, uchar* argv[])
{

	namedWindow("src", WINDOW_AUTOSIZE);
	namedWindow("dst", WINDOW_AUTOSIZE);


	Mat src, dst;

	src = imread("E:\\1.png");
	CV_Assert(!src.empty());

	int spatialRad = 10;
	int colorRad = 10;
	int maxPryLevel = 1;

	//��ȻcreateTrackbar�����Ĳ���onChange����Ҫ����2��������ʽΪonChange(int,void*)
	//����������ϵͳ��Ӧ��������ʹ��createTrackbar����ʱ������õĺ������Բ���д����������
	//���Ŷ�����д����������ú�����ʵ�ֹ����л�����Ҫ����(int,void*)2����������
	createTrackbar("spatialRad", "dst", &spatialRad, 80);
	createTrackbar("colorRad", "dst", &colorRad, 60);
	createTrackbar("maxPryLevel", "dst", &maxPryLevel, 5);
	bool bRet = false;
	ImageSegment ism(spatialRad, colorRad, maxPryLevel); 
	bRet = ism.imageSegment(src, dst, IMG_SEGMENT_MEANSHIFT, spatialRad, colorRad, maxPryLevel);

	

	//    meanshift_seg(0,0);

	imshow("src", src);
	/*char c=(char)waitKey();
	if(27==c)
	return 0;*/
	imshow("dst", dst);
	imshow("flood", src);
	waitKey();//���޵ȴ��û�������Ӧ
	//    while(1);//���ﲻ����while(1)��ԭ������Ҫ�ȴ��û��Ľ�������while(1)û�иù��ܡ���Ȼ2�߶������޵ȴ������á�
	return 0;
}
