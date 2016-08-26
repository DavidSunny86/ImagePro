#include "ImageTextFeature.hpp"


ImageTextFeature::ImageTextFeature(Mat& srcMat) :m_srcMat(srcMat)
{
	
}



int ImageTextFeature::sheetLBPFeature(const Mat& src, const Mat& dst, int mode, int thr)
{
	if (src.empty() || dst.empty())
	{
		return  LP_SHEET_INPUT_ERR;
	}

	int width = src->cols;
	int height = src->rows;
	uchar table[256];
	lbp59table(table);
	for (int j = 1; j < width - 1; j++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			uchar neighborhood[8] = { 0 };
			neighborhood[7] = CV_MAT_ELEM(src, uchar, i - 1, j - 1);
			neighborhood[6] = CV_MAT_ELEM(src, uchar, i - 1, j);
			neighborhood[5] = CV_MAT_ELEM(src, uchar, i - 1, j + 1);
			neighborhood[4] = CV_MAT_ELEM(src, uchar, i, j + 1);
			neighborhood[3] = CV_MAT_ELEM(src, uchar, i + 1, j + 1);
			neighborhood[2] = CV_MAT_ELEM(src, uchar, i + 1, j);
			neighborhood[1] = CV_MAT_ELEM(src, uchar, i + 1, j - 1);
			neighborhood[0] = CV_MAT_ELEM(src, uchar, i, j - 1);
			uchar center = CV_MAT_ELEM(src, uchar, i, j);
			uchar temp = 0;

			if (LBP_UNIFORM_INVARIANCE == mode)
			{

				for (int k = 0; k < 8; k++)
				{
					temp += (neighborhood[k] >= center) << k;
				}
				CV_MAT_ELEM(dst, uchar, i, j) = table[temp];
			}
			else if (LBP_GRAY_INVARIANCE == mode)
			{

				for (int k = 0; k < 8; k++)
				{
					temp += (neighborhood[k] >= center) << k;
				}
				CV_MAT_ELEM(dst, uchar, i, j) = temp;
			}
			else if (LBP_ROTATION_INVARIANCE == mode)
			{
				uchar minTemp = 255;
				for (int l = 0; l < 8; l++)
				{
					for (int k = 0; k < 8; k++)
					{
						int index = k + l < 8 ? (k + l) : (k + l - 8);
						temp += ((uchar)(neighborhood[index] >= center)) << k;
					}

					if (temp < minTemp)
					{
						minTemp = temp;
					}
				}
				CV_MAT_ELEM(dst, uchar, i, j) = minTemp;
			}
			else if (LBP_UNROTATION_INVARIANCE == mode)
			{
				uchar minTemp = 255;
				for (int l = 0; l < 8; l++)
				{
					for (int k = 0; k < 8; k++)
					{
						int index = k + l < 8 ? (k + l) : (k + l - 8);
						temp += (neighborhood[index] >= center) << k;
					}

					if (temp < minTemp)
					{
						minTemp = temp;
					}
				}
				CV_MAT_ELEM(dst, uchar, i, j) = table[minTemp];
			}

			else if (LBP_ROTATION_INVARIANCE_THR == mode)
			{
				uchar minTemp = 255;
				for (int l = 0; l < 8; l++)
				{
					for (int k = 0; k < 8; k++)
					{
						int index = k + l < 8 ? (k + l) : (k + l - 8);
						temp += ((uchar)(neighborhood[index] >= center + thr)) << k;
					}

					if (temp < minTemp)
					{
						minTemp = temp;
					}
				}
				CV_MAT_ELEM(dst, uchar, i, j) = minTemp;
			}
			else
			{
				break;
			}
		}
	}
	return LP_SHEET_OK;
}


int ImageTextFeature::getHopCount(uchar i)
{
	int a[8] = { 0 };
	int k = 7;
	int cnt = 0;
	while (i)
	{
		a[k] = i & 1;
		i >>= 1;
		--k;
	}
	for (int k = 0; k < 8; ++k)
	{
		if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
		{
			++cnt;
		}
	}
	return cnt;
}
void ImageTextFeature::lbp59table(uchar* table)
{
	memset(table, 0, 256);
	uchar temp = 1;
	for (int i = 0; i < 256; ++i)
	{
		if (getHopCount(i) <= 2)
		{
			table[i] = temp;
			temp++;
		}
	}
}

int ImageTextFeature::calcGLCM(IplImage* bWavelet, int angleDirection, double* featureVector)
{
	int i, j;
	int width, height;

	if (NULL == bWavelet)
		return 1;

	width = bWavelet->width;

	int * glcm = new int[GLCM_CLASS * GLCM_CLASS];
	int * histImage = new int[width * height];

	if (NULL == glcm || NULL == histImage)
		return 2;

	//灰度等级化---分GLCM_CLASS个等级  
	uchar *data = (uchar*)bWavelet->imageData;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			histImage[i * width + j] = (int)(data[bWavelet->widthStep * i + j] * GLCM_CLASS / 256);
		}
	}

	//初始化共生矩阵  
	for (i = 0; i < GLCM_CLASS; i++)
		for (j = 0; j < GLCM_CLASS; j++)
			glcm[i * GLCM_CLASS + j] = 0;

	//计算灰度共生矩阵  
	int w, k, l;
	//水平方向  
	if (angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width)
				{
					k = histImage[i * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width)
				{
					k = histImage[i * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//垂直方向  
	else if (angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//对角方向  
	else if (angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];

				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}

	//计算特征值  
	double entropy = 0, energy = 0, contrast = 0, homogenity = 0;
	for (i = 0; i < GLCM_CLASS; i++)
	{
		for (j = 0; j < GLCM_CLASS; j++)
		{
			//熵  
			if (glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			//能量  
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			//对比度  
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			//一致性  
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}
	//返回特征值  
	i = 0;
	featureVector[i++] = entropy;
	featureVector[i++] = energy;
	featureVector[i++] = contrast;
	featureVector[i++] = homogenity;

	delete[] glcm;
	delete[] histImage;
	return 0;
}