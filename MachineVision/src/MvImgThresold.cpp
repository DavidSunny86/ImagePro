#include "mvImgThresold.h"


ImageThresold::ImageThresold()
{
}

int ImageThresold::getThresoldImage(const Mat& src, Mat& thrMat,int thresoldMathod)
{
	if (src.empty())
	{
		return -1;
	}

	int iRet = 0;
	Mat dstTmp;
	switch (thresoldMathod)
	{
	case IMAGE_THRESOLD_GLOBAL:
	{
		dstTmp = globalThresold(src, calculateBestGlobalThr(src));
	}
	break;
	case IMAGE_THRESOLD_LOCAL:
	{
		dstTmp = localThresold(const Mat& src)
	}
	break;
	case IMAGE_THRESOLD_MAXENTROPY:
	{
		dstTmp =maxEntropy(src);
	}
	break;
	default:
		break;
	}
	thrMat = dstTmp;

	return 0;

}

int ImageThresold::thresholdOtsu(const Mat& src)
{
	//histogram
	float histogram[256] = { 0 };
	for (int j = 0; j < src.rows; j++)
	{
		uchar* data = src.ptr<uchar>(j);
		for (int i = 0; i < src.cols; i++)
		{
			histogram[data[i]]++;
		}
	}

	//normalize histogram
	int size = height*width;
	for (int i = 0; i < 256; i++)
	{
		histogram[i] = histogram[i] / size;
	}

	//average pixel value
	float avgValue = 0;
	for (int i = 0; i < 256; i++)
	{
		avgValue += i*histogram[i];
	}

	int threshold;
	float maxVariance = 0;
	float w = 0, u = 0;
	for (int i = 0; i < 256; i++)
	{
		w += histogram[i];
		u += i*histogram[i];

		float t = avgValue*w - u;
		float variance = t*t / (w*(1 - w));
		if (variance > maxVariance) {
			maxVariance = variance;
			threshold = i;
		}
	}

	return threshold;
}


cv::Mat ImageThresold::globalThresold(const Mat& src)
{

}

cv::Mat ImageThresold::localThresold(const Mat& src)
{

}

int ImageThresold::ostuThresold(const Mat& src,const Mat thrMat)
{
	if (src.empty())
		return -1;

	Mat dstTmp;

	int ostuThr = ostuThresold(src);
	threshold(src, dstTmp, ostuThr, 255, cv::THRESH_BINARY);
	thrMat = dstTmp;
	return 0;

}

cv::Mat ImageThresold::maxEntropy(const Mat& src)
{
	if (src.empty())
		return src;

	CvHistogram * hist = cvCreateHist(1, &HistBins, CV_HIST_ARRAY, HistRange);//����һ��ָ���ߴ��ֱ��ͼ
	//�������壺ֱ��ͼ������ά����ֱ��ͼά���ߴ�����顢ֱ��ͼ�ı�ʾ��ʽ�����鷶Χ���顢��һ����־
	cvCalcHist(&src, hist);//����ֱ��ͼ
	double maxentropy = -1.0;
	int max_index = -1;
	// ѭ������ÿ���ָ�㣬Ѱ�ҵ�������ֵ�ָ��
	for (int i = 0; i < HistBins; i++)
	{
		double cur_entropy = caculateCurrentEntropy(hist, i, object) + caculateCurrentEntropy(hist, i, back);
		if (cur_entropy > maxentropy)
		{
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	cout << "The Threshold of this Image in MaxEntropy is:" << max_index << endl;
	threshold(src, dst, (double)max_index, 255, THRESH_BINARY);
	cvReleaseHist(&hist);

	return dst;

}

double ImageThresold::caculateCurrentEntropy(CvHistogram * Hist, int curThr, EntropyState state)
{
	int start, end;
	int total = 0;
	double cur_entropy = 0.0;
	if (state == back)
	{
		start = 0;
		end = curThr;
	}
	else
	{
		start = curThr;
		end = 256;
	}
	for (int i = start; i<end; i++)
	{
		total += (int)cvQueryHistValue_1D(Hist, i);//��ѯֱ�����ֵ P304
	}
	for (int j = start; j<end; j++)
	{
		if ((int)cvQueryHistValue_1D(Hist, j) == 0)
			continue;
		double percentage = cvQueryHistValue_1D(Hist, j) / total;
		/*�صĶ��幫ʽ*/
		cur_entropy += -percentage*logf(percentage);
		/*����̩��չʽȥ���ߴ���õ����صĽ��Ƽ��㹫ʽ
		cur_entropy += percentage*percentage;*/
	}
	return cur_entropy;

}

int ImageThresold::calculateBestGlobalThr(const Mat& src)
{
	int hist[256] = { 0 };
	for (int j = 0; j < src.rows; j++)
	{
		uchar* data = src.ptr<uchar>(j);
		for (int i = 0; i < src.cols; i++)
		{
			hist[data[i]]++;
		}
	}

	int i, t, t1, t2, k1, k2;
	double u, u1, u2;
	t = 0;
	u = 0;
	for (i = 0; i < 256; i++)
	{
		t += hist[i];
		u += i*hist[i];
	}
	k2 = (int)(u / t); // ����˷�Χ�Ҷȵ�ƽ��ֵ
	do
	{
		k1 = k2;
		t1 = 0;
		u1 = 0;
		for (i = start; i <= k1; i++)
		{ // ����ͻҶ�����ۼӺ�
			t1 += hist[i];
			u1 += i*hist[i];
		}
		t2 = t - t1;
		u2 = u - u1;
		if (t1)
			u1 = u1 / t1; // ����ͻҶ����ƽ��ֵ
		else
			u1 = 0;
		if (t2)
			u2 = u2 / t2; // ����߻Ҷ����ƽ��ֵ
		else
			u2 = 0;
		k2 = (int)((u1 + u2) / 2); // �õ��µ���ֵ����ֵ
	} while (k1 != k2); // ����δ�ȶ�������

	return k1; // ������ֵ
}


void ImageThresold::bersenLocalThreshold(const Mat& img, Mat& mask, Mat kernelMat, int localContrastThre, int grayThre)
{
    Mat backgroundMat(img.rows, img.cols, img.type());
    backgroundMat.setTo(grayThre);
    bersenLocalThreshold(img, mask, kernelMat, localContrastThre, backgroundMat);
}

void ImageThresold::bersenLocalThreshold(const Mat& img, Mat& mask, int ksize, int localContrastThre, int grayThre)
{
    Mat kernelMat = Mat::ones(ksize, ksize, CV_8UC1);
    bersenLocalThreshold(img, mask, kernelMat, localContrastThre, grayThre);
}

void ImageThresold::bersenLocalYThreshold(const Mat& img, Mat& mask, int ksize, int localContrastThre, int grayThre)
{
    Mat kernelMat = Mat::ones(ksize, 1, CV_8UC1);
    bersenLocalThreshold(img, mask, kernelMat, localContrastThre, grayThre);
}

void ImageThresold::bersenLocalYThreshold(const Mat& img, Mat& mask, int ksize, int localContrastThre, const Mat& backgroundMat)
{
    Mat kernelMat = Mat::ones(ksize, 1, CV_8UC1);
    bersenLocalThreshold(img, mask, kernelMat, localContrastThre, backgroundMat);
}

void ImageThresold::bersenLocalXThreshold(const Mat& img, Mat& mask, int ksize, int localContrastThre, int grayThre)
{
    Mat kernelMat = Mat::ones(1, ksize, CV_8UC1);
    bersenLocalThreshold(img, mask, kernelMat, localContrastThre, grayThre);
}

void ImageThresold::bersenLocalXThreshold(const Mat& img, Mat& mask, int ksize, int localContrastThre, const Mat& backgroundMat)
{
    Mat kernelMat = Mat::ones(1, ksize, CV_8UC1);
    bersenLocalThreshold(img, mask, kernelMat, localContrastThre, backgroundMat);
}

void ImageThresold::bersenLocalThreshold(const Mat& img, Mat& mask, const Mat& kernelMat, int localContrastThre, const Mat& backgroundMat)
{
    Mat maxMat, minMat;
    dilate(img, maxMat, kernelMat);
    erode(img, minMat, kernelMat);

    Mat localContrastMat = maxMat - minMat;
    Mat midGrayMat = (maxMat + minMat) / 2;

    Mat localContrastMask = localContrastMat < localContrastThre;
    Mat midGrayMask = midGrayMat >= backgroundMat;

    mask = localContrastMask & midGrayMask;

    Mat invLocalContrastMask = ~localContrastMask;
    mask |= invLocalContrastMask & (img > midGrayMat);
	return ;
}



ImageThresold::~ImageThresold()
{
}
