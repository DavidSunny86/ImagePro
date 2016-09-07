#include "MachineLearn.h"


MachineLearn::MachineLearn()
{
}


double g_MaxSimValue = 0.0;


MachineLearn::MachineLearn()
{
}

MachineLearn::MachineLearn(String  clsModeSavePath, int iMlMode)
{
	loadMlCls(clsModeSavePath, iMlMode);
	m_iMlModel = iMlMode;
	m_bIsLoadClsModel = true;
	return;
}

MachineLearn::MachineLearn(String trainSamplePath, String clsModeSavePath, int iMlMode, int param0, int param1)
{
	int iRet = 0;
	if (trainSamplePath.empty() || clsModeSavePath.empty())
	{
		return;
	}

	iRet = buildMlClassfierAndSave(trainSamplePath, clsModeSavePath, iMlMode, param0, param1);
	if (iRet != 0)
	{
		return;
	}

	m_bIsTrainSample = true;
	return;

}


list<Ml_Base_Feature> MachineLearn::computerFeature(String dataPath, int mode)
{
	list<Ml_Base_Feature> Featurelist;

	return Featurelist;
}

IplImage* MachineLearn::getImgFromStyleNo(String StyleNo)
{
	if (StyleNo())
	{
		return NULL;
	}
}


String MachineLearn::getClsName(int iMlModel)
{
	String sClsName;
	switch (iMlModel)
	{
	case 0:
		sClsName = "ML_METHOD_ADABOOST";
		break;
	case 1:
		sClsName = "ML_METHOD_RTREES";
		break;
	case 2:
		sClsName = "ML_METHOD_MLP";
		break;
	case 3:
		sClsName = "ML_METHOD_KNN";
		break;
	case 4:
		sClsName = "ML_METHOD_SVM";
		break;
	case 5:
		sClsName = "ML_METHOD_NBAYES";
		break;
	default:
		break;
	}

	return sClsName;
}



int MachineLearn::buildMlClassfierSampleData(String trainPath, Mat* data, Mat* responsees, int iReducedDimMode)
{
	list <Ml_Base_Feature> allFeature = computerFeature(trainPath, computerFeatureMode);
	int iRet = 0;
	int size = allFeature.size();
	vector<int> responsees_temp;
	Mat matTemp;

	if (size == 0)
	{
		return  -1;
	}

	for (auto temp : allFeature)
	{
		responsees_temp.push_back(temp.iLabel);
		matTemp.push_back(temp);
	}

	Mat(matTemp).copyTo(*data);

	Mat(responsees_temp).copyTo(*responsees);

	return 0;
}

//准备训练数据
Ptr<TrainData> MachineLearn::prepareTrainData(const Mat& data, const Mat& responses, int ntrain_samples)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(cv::Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

//设置迭代条件
inline TermCriteria MachineLearn::setIterCondition(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

//分类预测
void MachineLearn::testAndSaveClassifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses, int iMlModel,
	float ntrain_samples, int rdelta, char* resultPath, char* param1, char* param2)
{
	int i, nsamples_all = data.rows;
	int trainNum = nsamples_all*ntrain_samples;
	double train_hr = 0;
	double test_hr = 0;
	ML_RPE_CLS_INFO PreCls[ML_CLASS_UNREV + 1] = { 0 };

	for (i = 0; i < ML_CLASS_UNREV + 1; i++)
	{
		PreCls[i] = { 0 };
	}

	String mlStyle = getClsName(iMlModel);
	String sParsm1 = param1;
	QFile sumfile(resultPath);
	if (!sumfile.exists())
	{
		sumfile.open(QIODevice::WriteOnly);
		sumfile.close();
	}
	String modelSavePath = resultPath;
	modelSavePath = modelSavePath + "/../" + mlStyle + sParsm1 + ".XML";
	model->save(modelSavePath.c_str());
	sumfile.open(QIODevice::WriteOnly | QIODevice::Append);
	QTextStream outSum(&sumfile);
	int iFirstFlag = 0;

	outSum << qSetFieldWidth(10) << left << mlStyle.c_str() << "  Save Result!!" << "\r\n";

	// compute prediction error on train and test data
	if (iMlModel == LP_ML_ADABOOST)
	{
		int var_count = data.cols;
		int k, j, i;

		Mat temp_sample(1, var_count + 1, CV_32F);
		float* tptr = temp_sample.ptr<float>();

		for (i = 0; i < nsamples_all; i++)
		{
			int best_class = 0;
			double max_sum = -DBL_MAX;
			const float* ptr = data.ptr<float>(i);
			for (k = 0; k < var_count; k++)
				tptr[k] = ptr[k];

			for (j = 0; j < class_count; j++)
			{
				tptr[var_count] = (float)j;
				float s = model->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);
				if (max_sum < s)
				{
					max_sum = s;
					best_class = j;
				}
			}

			double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ? 1 : 0;

			PreCls[responses.at<int>(i)].srcClsStyle = getClsReTransefer(responses.at<int>(i));
			PreCls[responses.at<int>(i)].sameStyleNum++;

			if (i < trainNum)
			{
				train_hr += r;
				PreCls[responses.at<int>(i)].iTrainNum++;
				if (r > 0.5)
				{
					PreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getClsReTransefer(responses.at<int>(i))<<" pre cls"<< getClsReTransefer((int)best_class) << "\r\n";*/
			}
			else
			{
				PreCls[responses.at<int>(i)].iTestNum++;
				if (r > 0.5)
				{
					PreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getClsReTransefer(responses.at<int>(i)) << " pre cls" << getClsReTransefer((int)best_class) << "\r\n";
				test_hr += r;
			}

		}
	}
	else
	{
		for (i = 0; i < nsamples_all; i++)
		{
			Mat sample = data.row(i);

			float r = model->predict(sample);
			float comR = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

			PreCls[responses.at<int>(i)].srcClsStyle = getClsReTransefer(responses.at<int>(i));
			PreCls[responses.at<int>(i)].sameStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i <= trainNum)
			{
				PreCls[responses.at<int>(i)].iTrainNum++;
				if (comR > 0.5)
				{
					PreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getClsReTransefer(responses.at<int>(i)) << " pre cls" << getClsReTransefer((int)r) << "\r\n";
				train_hr += comR;
			}
			else
			{
				PreCls[responses.at<int>(i)].iTestNum++;
				if (comR > 0.5)
				{
					PreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getClsReTransefer(responses.at<int>(i)) << " pre cls" << getClsReTransefer((int)r) << "\r\n";*/
				test_hr += comR;
			}
		}
	}

	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;
	test_hr = (nsamples_all - trainNum) > 0 ? test_hr / (nsamples_all - trainNum) : 1.;

	outSum << qSetFieldWidth(10) << left << mlStyle.c_str() << qSetFieldWidth(10) << left << param1 <<
		qSetFieldWidth(10) << left << param2 << "trainNum" << trainNum << qSetFieldWidth(1) << " TrainClsRatio =" << train_hr <<
		qSetFieldWidth(10) << " TestClsRatio =" << test_hr << "\r\n";

	for (int cls = 0; cls < LP__CLASS_UNREV; cls++)
	{

		double trainHr = PreCls[cls].iTrainNum > 0 ? (PreCls[cls].preTrainClsCurrentNum*1.0) / (PreCls[cls].iTrainNum*1.0) : 0.0f;
		double testHr = PreCls[cls].iTestNum > 0 ? (PreCls[cls].preTestClsCurrentNum*1.0) / (PreCls[cls].iTestNum*1.0) : 0.0f;

		outSum << qSetFieldWidth(10) << left << PreCls[cls].srcClsStyle << qSetFieldWidth(10) << left << "All  Num" << PreCls[cls].sameStyleNum << qSetFieldWidth(10) << left << "trainNum" << PreCls[cls].iTrainNum << qSetFieldWidth(1) << " TrainClsRatio =" << trainHr <<
			qSetFieldWidth(10) << " TestClsRatio =" << testHr << "\r\n";
	}


	outSum.flush();
	sumfile.close();
}


//分类预测
void MachineLearn::testAndSaveClassifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses, list<Ml_Base_Feature> srcSampleData, int iMlModel, float ntrain_samples, int rdelta, char* resultPath, char* param1, char* param2)
{
	int i, nsamples_all = data.rows;
	int trainNum = nsamples_all*ntrain_samples;
	double train_hr = 0;
	double test_hr = 0;
	ML_RPE_CLS_INFO PreCls[ML_CLASS_UNREV + 1] = { 0 };
	CLS_PREDICT_PRO ClsPrePro[ML_CLASS_UNREV + 1];

	for (i = 0; i < ML_CLASS_UNREV + 1; i++)
	{
		PreCls[i] = { 0 };
		ClsPrePro[i] = { 0 };
	}

	String mlStyle = getClsName(iMlModel);
	String sParsm0 = param1;
	String sParsm1 = param2;

	String subFilePath = resultPath;
	subFilePath = subFilePath + "/../ClS_Sub.txt";
	QFile clsfile(subFilePath.c_str());
	if (!clsfile.exists())
	{
		clsfile.open(QIODevice::WriteOnly);
		clsfile.close();
	}

	QFile sumfile(resultPath);
	if (!sumfile.exists())
	{
		sumfile.open(QIODevice::WriteOnly);
		sumfile.close();
	}

	sumfile.open(QIODevice::WriteOnly | QIODevice::Append);
	QTextStream outSum(&sumfile);

	clsfile.open(QIODevice::WriteOnly | QIODevice::Append);
	QTextStream clsStr(&clsfile);

	int iFirstFlag = 0;


	String sSaveBasePath = resultPath;
	sSaveBasePath = sSaveBasePath + "/../" + mlStyle + sParsm0 + "_" + sParsm1;
	QDir qdirTcFileName;
	bool bFolder = qdirTcFileName.mkpath(String::fromStdString(sSaveBasePath));

	if (srcSampleData.size() != responses.rows)
	{
		return;
	}

	// compute prediction error on train and test data
	if (iMlModel == ML_METHOD_ADABOOST)
	{
		int var_count = data.cols;
		int k, j, i;

		Mat temp_sample(1, var_count + 1, CV_32F);
		float* tptr = temp_sample.ptr<float>();

		for (i = 0; i < nsamples_all; i++)
		{
			int best_class = 0;
			double max_sum = -DBL_MAX;
			const float* ptr = data.ptr<float>(i);
			for (k = 0; k < var_count; k++)
				tptr[k] = ptr[k];

			for (j = 0; j < class_count; j++)
			{
				tptr[var_count] = (float)j;
				float s = model->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);
				if (max_sum < s)
				{
					max_sum = s;
					best_class = j;
				}
			}

			double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ? 1 : 0;

			PreCls[responses.at<int>(i)].srcClsStyle = getClsReTransefer(responses.at<int>(i));
			PreCls[responses.at<int>(i)].sameStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i < trainNum)
			{
				train_hr += r;
				PreCls[responses.at<int>(i)].iTrainNum++;
				if (r > 0.5)
				{
					PreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}

				PreCls[responses.at<int>(i)].preTrainClsPre[best_class]++;
				/*		outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getClsReTransefer(responses.at<int>(i))<<" pre cls"<< getClsReTransefer((int)best_class) << "\r\n";*/
			}
			else
			{
				PreCls[responses.at<int>(i)].iTestNum++;
				PreCls[responses.at<int>(i)].preTestClsPre[best_class]++;
				if (r > 0.5)
				{
					PreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				else
				{
					String StyleNo = srcSampleData.at(i).sLable;
					IplImage* pCurImg = getImgFromStyleNo(srcSampleData.at(i).sLable);
					String preCls = getClsReTransefer(best_class);
					String ImgSavePath = sSaveBasePath + "/" + StyleNo.toStdString() + "_" + preCls.toStdString() + ".png";
					cvSaveImage(ImgSavePath.c_str(), pCurImg);
					cvReleaseImage(&pCurImg);

					ClsPrePro[responses.at<int>(i)].SrcCls = responses.at<int>(i);
					ClsPrePro[responses.at<int>(i)].PreCls[best_class]++;

				}

				//outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getClsReTransefer(responses.at<int>(i)) << " pre cls" << getClsReTransefer((int)best_class) << "\r\n";
				test_hr += r;
			}

		}
	}
	else
	{
		for (i = 0; i < nsamples_all; i++)
		{
			Mat sample = data.row(i);

			float r = model->predict(sample);
			float comR = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

			PreCls[responses.at<int>(i)].srcClsStyle = getClsReTransefer(responses.at<int>(i));
			PreCls[responses.at<int>(i)].sameStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i <= trainNum)
			{
				PreCls[responses.at<int>(i)].iTrainNum++;
				PreCls[responses.at<int>(i)].preTrainClsPre[(int)r]++;
				if (comR > 0.5)
				{
					PreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getClsReTransefer(responses.at<int>(i)) << " pre cls" << getClsReTransefer((int)r) << "\r\n";
				train_hr += comR;
			}
			else
			{
				PreCls[responses.at<int>(i)].iTestNum++;
				PreCls[responses.at<int>(i)].preTestClsPre[(int)r]++;
				if (comR > 0.5)
				{
					PreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getClsReTransefer(responses.at<int>(i)) << " pre cls" << getClsReTransefer((int)r) << "\r\n";*/

				else
				{
					String StyleNo = srcSampleData.at(i).sLable;
					IplImage* pCurImg = getImgFromStyleNo(srcSampleData.at(i).sLable);
					String preCls = getClsReTransefer(r);
					String ImgSavePath = sSaveBasePath + "/" + StyleNo.toStdString() + "_" + preCls.toStdString() + ".png";
					cvSaveImage(ImgSavePath.c_str(), pCurImg);
					cvReleaseImage(&pCurImg);
				}

				test_hr += comR;
			}
		}
	}

	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;
	test_hr = (nsamples_all - trainNum) > 0 ? test_hr / (nsamples_all - trainNum) : 1.;

	outSum << qSetFieldWidth(15) << left << mlStyle.c_str() << qSetFieldWidth(10) << left << param1 <<
		qSetFieldWidth(10) << left << param2 << qSetFieldWidth(10) << left << "trainNum" << qSetFieldWidth(5) <<
		trainNum << qSetFieldWidth(15) << " TrainClsRatio =" << qSetFieldWidth(10) << train_hr << qSetFieldWidth(15)
		<< " TestClsRatio =" << qSetFieldWidth(10) << test_hr << "\r\n";

	outSum.flush();
	sumfile.close();

	for (int cls = 0; cls < LP_CLASS_UNREV; cls++)
	{

		double trainHr = PreCls[cls].iTrainNum > 0 ? (PreCls[cls].preTrainClsCurrentNum*1.0) / (PreCls[cls].iTrainNum*1.0) : 0.0f;
		double testHr = PreCls[cls].iTestNum > 0 ? (PreCls[cls].preTestClsCurrentNum*1.0) / (PreCls[cls].iTestNum*1.0) : 0.0f;

		double train0 = PreCls[cls].preTrainClsPre[0] * 1.0 / PreCls[cls].iTrainNum*1.0;
		double train1 = PreCls[cls].preTrainClsPre[1] * 1.0 / PreCls[cls].iTrainNum*1.0;
		double train2 = PreCls[cls].preTrainClsPre[2] * 1.0 / PreCls[cls].iTrainNum*1.0;

		double test0 = PreCls[cls].preTestClsPre[0] * 1.0 / PreCls[cls].iTestNum*1.0;
		double test1 = PreCls[cls].preTestClsPre[1] * 1.0 / PreCls[cls].iTestNum*1.0;
		double test2 = PreCls[cls].preTestClsPre[2] * 1.0 / PreCls[cls].iTestNum*1.0;


		clsStr << qSetFieldWidth(10) << left << PreCls[cls].srcClsStyle << qSetFieldWidth(10) << left << "All  Num" << qSetFieldWidth(5) << PreCls[cls].sameStyleNum << qSetFieldWidth(10) << left << "trainNum" << qSetFieldWidth(5) << PreCls[cls].iTrainNum << qSetFieldWidth(15) << " TrainClsRatio =" <<
			qSetFieldWidth(10) << trainHr << qSetFieldWidth(15) << " TestClsRatio =" << qSetFieldWidth(10) << testHr << "Train  " <<
			qSetFieldWidth(1) << PreCls[cls].srcClsStyle << "-" << "E=" << qSetFieldWidth(10) << train0 <<
			qSetFieldWidth(1) << PreCls[cls].srcClsStyle << "-" << "F=" << qSetFieldWidth(10) << train1 <<
			qSetFieldWidth(1) << PreCls[cls].srcClsStyle << "-" << "S=" << qSetFieldWidth(10) << train2 <<
			"Test  " <<
			qSetFieldWidth(1) << PreCls[cls].srcClsStyle << "-" << "E=" << qSetFieldWidth(10) << test0 <<
			qSetFieldWidth(1) << PreCls[cls].srcClsStyle << "-" << "F=" << qSetFieldWidth(10) << test1 <<
			qSetFieldWidth(1) << PreCls[cls].srcClsStyle << "-" << "S=" << qSetFieldWidth(10) << test2 <<
			qSetFieldWidth(10)

			<< "\r\n";
	}

	clsStr.flush();
	clsfile.close();

	if (test_hr > g_MaxSimValue)
	{
		String modelSavePath = resultPath;
		modelSavePath = modelSavePath + "/../" + mlStyle + sParsm1 + ".XML";
		model->save(modelSavePath.c_str());

		g_MaxSimValue = test_hr;
	}
	return;

}



//随机树分类
Ptr<StatModel> MachineLearn::buildRtreesClassifier(Mat data, Mat  responses, int ntrain_samples, double maxDepth, double iter)
{

	Ptr<RTrees> model;
	Ptr<TrainData> tdata = prepareTrainData(data, responses, ntrain_samples);
	model = RTrees::create();
	model->setMaxDepth(maxDepth);
	model->setMinSampleCount(10);
	model->setRegressionAccuracy(0);
	model->setUseSurrogates(false);
	model->setMaxCategories(15);
	model->setPriors(Mat());
	model->setCalculateVarImportance(false);
	model->setTermCriteria(setIterCondition(iter, 0.01f));
	model->train(tdata);

	return model;
}

//adaboost分类
Ptr<StatModel> MachineLearn::buildAdaboostClassifier(Mat data, Mat  responses, int ntrain_samples, double param0, double param1)
{
	Mat weak_responses;
	int i, j, k;
	Ptr<Boost> model;

	int nsamples_all = data.rows;
	int var_count = data.cols;

	Mat new_data(ntrain_samples*class_count, var_count + 1, CV_32F);
	Mat new_responses(ntrain_samples*class_count, 1, CV_32S);

	for (i = 0; i < ntrain_samples; i++)
	{
		const float* data_row = data.ptr<float>(i);
		for (j = 0; j < class_count; j++)
		{
			float* new_data_row = (float*)new_data.ptr<float>(i*class_count + j);
			memcpy(new_data_row, data_row, var_count*sizeof(data_row[0]));
			new_data_row[var_count] = (float)j;
			new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j;
		}
	}

	Mat var_type(1, var_count + 2, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count + 1) = VAR_CATEGORICAL;

	Ptr<TrainData> tdata = TrainData::create(new_data, ROW_SAMPLE, new_responses,
		noArray(), noArray(), noArray(), var_type);

	model = Boost::create();
	model->setBoostType(Boost::GENTLE);
	model->setWeakCount(param0);
	model->setWeightTrimRate(0.95);
	model->setMaxDepth(param1);
	model->setUseSurrogates(false);
	model->train(tdata);

	return model;
}

//多层感知机分类（ANN）
Ptr<StatModel> MachineLearn::buildMlpClassifier(Mat data, Mat  responses, int ntrain_samples, double param0, double param1)
{
	//read_num_class_data(data_filename, 16, &data, &responses);
	Ptr<ANN_MLP> model;
	Mat train_data = data.rowRange(0, ntrain_samples);
	Mat train_responses = Mat::zeros(ntrain_samples, class_count, CV_32F);

	// 1. unroll the responses
	for (int i = 0; i < ntrain_samples; i++)
	{
		int cls_label = responses.at<int>(i);
		train_responses.at<float>(i, cls_label) = 1.f;
	}

	// 2. train classifier
	int layer_sz[] = { data.cols, 100, 100, class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = param1;
	int max_iter = param0;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(setIterCondition(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
	return model;
}


//贝叶斯分类
Ptr<StatModel> MachineLearn::buildNbayesClassifier(Mat data, Mat  responses, int ntrain_samples)
{
	Ptr<NormalBayesClassifier> model;
	Ptr<TrainData> tdata = prepareTrainData(data, responses, ntrain_samples);
	model = NormalBayesClassifier::create();
	model->train(tdata);

	return model;
}

Ptr<StatModel>  MachineLearn::buildKnnClassifier(Mat data, Mat  responses, int ntrain_samples, int K)
{
	Ptr<TrainData> tdata = prepareTrainData(data, responses, ntrain_samples);
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tdata);

	return model;
}

//svm分类
Ptr<StatModel> MachineLearn::buildSvmClassifier(Mat data, Mat  responses, int ntrain_samples)
{
	Ptr<SVM> model;
	Ptr<TrainData> tdata = prepareTrainData(data, responses, ntrain_samples);
	model = SVM::create();
	model->setType(SVM::C_SVC);
	model->setKernel(SVM::RBF);
	model->setC(1);
	model->train(tdata);
	return model;
}


int MachineLearn::predictCls(const Ptr<StatModel>& model, const Mat pendSample, int iMlModel)
{
	if (model->empty())
	{
		return  -1;
	}

	if (iMlModel == ML_METHOD_ADABOOST)
	{
		int var_count = pendSample.cols;
		int k, j, best_class;

		Mat temp_sample(1, var_count + 1, CV_32F);
		float* tptr = temp_sample.ptr<float>();

		double max_sum = -DBL_MAX;
		const float* ptr = pendSample.ptr<float>(0);
		for (k = 0; k < var_count; k++)
			tptr[k] = ptr[k];

		for (j = 0; j < class_count; j++)
		{
			tptr[var_count] = (float)j;
			float s = model->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);
			if (max_sum < s)
			{
				max_sum = s;
				best_class = j;
			}
		}
		return best_class;
	}
	else
	{
		float r = model->predict(pendSample);
		return (int)r;
	}
	return -1;
}


int MachineLearn::predictCls(const Ptr<StatModel>& model, lppca pca, const Mat pendSample, int iMlModel)
{
	Mat pcaTemp;
	int iRet = 0;

	if (!pca.m_bIsBuildPca)
	{
		return  -1;
	}

	iRet = pca.getReductDimMat(pendSample, &pcaTemp);
	int iCls = predictCls(model, pendSample, iMlModel);

	if (iRet != 0 || iCls < 0)
	{
		return -1;
	}
	return  iCls;
}

int  MachineLearn::predictClsVote(vector<CLS_PREDICT_VOTE> PredictVote, int strictMode)
{
	int Precls[ML_CLASS_UNREV] = { 0 };
	bool bIsVote = false;
	int BoostPre = ML_CLASS_UNREV;
	for (int i = 0; i < PredictVote.size(); i++)
	{
		int Cls = PredictVote.at(i).PreCls;
		Precls[Cls]++;

		if (PredictVote.at(i).mlMode == LP_ML_ADABOOST)
			BoostPre = Cls;
	}

	// 严格投票，只有全票通过才归为一类，否则就归为简单类
	if (strictMode)
	{
		for (int i = 0; i < LP__CLASS_UNREV; i++)
		{
			if (Precls[i] == (LP__CLASS_UNREV - 1))
			{
				bIsVote = true;
				return i;
			}
		}

		if (bIsVote == false)
		{
			return LP__CLASS_COARSE_GRAIN;
		}
	}
	else
	{
		for (int i = 0; i < LP__CLASS_UNREV; i++)
		{
			if (Precls[i] > 1)
			{
				bIsVote = true;
				return i;
			}
		}

		if (!bIsVote)
		{
			return  LP__CLASS_COARSE_GRAIN;
		}
	}

	return LP__CLASS_COARSE_GRAIN;
}

/*针对不同种类进行判断 */
int MachineLearn::predictClsVote(vector<CLS_PREDICT_VOTE> PredictVote, double* ClsPrePro)
{
	int Precls[LP__CLASS_UNREV] = { 0 };
	bool bIsVote = false;
	int BoostPre = LP__CLASS_UNREV;

	for (int i = 0; i < PredictVote.size(); i++)
	{
		int Cls = PredictVote.at(i).PreCls;
		Precls[Cls]++;
	}

	for (int i = 0; i < LP__CLASS_UNREV; i++)
	{
		ClsPrePro[i] = Precls[i] * 1.0 / (LP__CLASS_UNREV - 1)*1.0;
	}

	return 0;
}


int MachineLearn::predictCls(const Mat pendSample, int iMlModel)
{
	if (m_bIsLoadClsModel)
	{
		if (iMlModel == LP_ML_ADABOOST)
		{
			int var_count = pendSample.cols;
			int k, j, best_class;

			Mat temp_sample(1, var_count + 1, CV_32F);
			float* tptr = temp_sample.ptr<float>();

			double max_sum = -DBL_MAX;
			const float* ptr = pendSample.ptr<float>(0);
			for (k = 0; k < var_count; k++)
				tptr[k] = ptr[k];

			for (j = 0; j < class_count; j++)
			{
				tptr[var_count] = (float)j;
				float s = m_pMlCls->predict(temp_sample, noArray(), StatModel::RAW_OUTPUT);
				if (max_sum < s)
				{
					max_sum = s;
					best_class = j;
				}
			}
			return best_class;
		}
		else
		{
			float r = m_pMlCls->predict(pendSample);
			return (int)r;
		}
	}
	return -1;
}

Ptr<StatModel> MachineLearn::buildClassifier(Mat data, Mat response, float ration, int mode, double param0, double param1)
{

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*ration);
	Ptr<StatModel> pClsModel;

	switch (mode)
	{
	case LP_ML_RTREES:
		pClsModel = buildRtreesClassifier(data, response, ntrain_samples, param0, param1);
		break;
	case LP_ML_ADABOOST:
		pClsModel = buildAdaboostClassifier(data, response, ntrain_samples, param0, param1);
		break;
	case LP_ML_MLP:
		pClsModel = buildMlpClassifier(data, response, ntrain_samples, param0, param1);
		break;
	case LP_ML_KNN:
		pClsModel = buildKnnClassifier(data, response, ntrain_samples, param0);
		break;
	case LP_ML_NBAYES:
		pClsModel = buildNbayesClassifier(data, response, ntrain_samples);
		break;
	case LP_ML_SVM:
		pClsModel = buildSvmClassifier(data, response, ntrain_samples);
		break;
	default:
		break;
	}

	return 	pClsModel;
}


Ptr<StatModel> MachineLearn::loadMlClassify(String clsPath, int iMlMode)
{

	switch (iMlMode)
	{
	case ML_METHOD_RTREES:
	{
		Ptr<RTrees> mlClsRt = RTrees::load<RTrees>(clsPath.c_str());
		int in = mlClsRt->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsRt;
			return mlClsRt;
		}
	}

	case ML_METHOD_ADABOOST:
	{
		Ptr<Boost> mlClsBoost = Boost::load<Boost>(clsPath.c_str());
		int in = mlClsBoost->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsBoost;
			return mlClsBoost;
		}
	}

	case ML_METHOD_MLP:
	{
		Ptr<ANN_MLP> mlClsAnn = ANN_MLP::load<ANN_MLP>(clsPath.c_str());
		int in = mlClsAnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsAnn;
			return mlClsAnn;
		}
	}

	case ML_METHOD_NBAYES:
	{
		Ptr<NormalBayesClassifier> mlClsNbayes = NormalBayesClassifier::load<NormalBayesClassifier>(clsPath.c_str());
		int in = mlClsNbayes->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsNbayes;
			return mlClsNbayes;
		}
	}

	case ML_METHOD_KNN:
	{
		Ptr<KNearest> mlClsKnn = KNearest::load<KNearest>(clsPath.c_str());
		int in = mlClsKnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsKnn;
			return mlClsKnn;
		}
	}

	case ML_METHOD_SVM:
	{
		Ptr<SVM> mlClsSvm = SVM::load<SVM>(clsPath.c_str());
		int in = mlClsSvm->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsSvm;
			return mlClsSvm;
		}
	}

	default:
		Ptr<StatModel> errCls;
		return errCls;
	}
}


void MachineLearn::loadMlCls(String clsPath, int iMlMode)
{

	switch (iMlMode)
	{
	case ML_METHOD_RTREES:
	{
		Ptr<RTrees> mlClsRt = RTrees::load<RTrees>(clsPath.c_str());

		int in = mlClsRt->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsRt;
			m_bIsLoadClsModel = true;
		}
		break;
	}

	case ML_METHOD_ADABOOST:
	{
		Ptr<Boost> mlClsBoost = Boost::load<Boost>(clsPath.c_str());
		int in = mlClsBoost->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsBoost;
			m_bIsLoadClsModel = true;
		}

		break;
	}

	case ML_METHOD_MLP:
	{
		Ptr<ANN_MLP> mlClsAnn = ANN_MLP::load<ANN_MLP>(clsPath.c_str());
		int in = mlClsAnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsAnn;
			m_bIsLoadClsModel = true;
		}
		break;
	}

	case ML_METHOD_NBAYES:
	{
		Ptr<NormalBayesClassifier> mlClsNbayes = NormalBayesClassifier::load<NormalBayesClassifier>(clsPath.c_str());
		int in = mlClsNbayes->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsNbayes;
			m_bIsLoadClsModel = true;
		}
		break;
	}

	case ML_METHOD_KNN:
	{
		Ptr<KNearest> mlClsKnn = KNearest::load<KNearest>(clsPath.c_str());
		int in = mlClsKnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsKnn;
			m_bIsLoadClsModel = true;
		}
		break;
	}

	case ML_METHOD_SVM:
	{
		Ptr<SVM> mlClsSvm = SVM::load<SVM>(clsPath.c_str());
		int in = mlClsSvm->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsSvm;
			m_bIsLoadClsModel = true;
		}
		break;
	}

	default:
		return;
	}

	return;
}

bool MachineLearn::isLoadMlCls()
{
	return m_bIsLoadClsModel;
}

void MachineLearn::buildMlClassfierTest(Mat data, Mat response, float ration, int iMlMode, char* resultPath, double params0)
{
	//建立模型
	Ptr<StatModel> pClsModel;
	char cParam[20];
	char cParam1[20];
	pClsModel = buildClassifier(data, response, ration, iMlMode, params0);

	//保存测试结果
	testAndSaveClassifier(pClsModel, data, response, iMlMode, ration, 0, resultPath, itoa(params0, cParam, 10), NULL);
	return;
}

void MachineLearn::buildMlClassfierTest(String trainPath, int iMlMode, char* testrResultPath, double param0, double param1)
{
	//建立训练样本
	Mat trainData;
	Mat trainLabel;

	for (int i = 1; i < 11; i++)
	{
		buildMlClassfierSampleData(trainPath, &trainData, &trainLabel, LP_ML_REDUCED_DIM_PCA);
		//建立模型
		buildMlClassfierTest(trainData, trainLabel, 0.8, iMlMode, testrResultPath, param0);
	}

	return;
}

list <Ml_Base_Feature> MachineLearn::get10FoldCrossValidation(list<Ml_Base_Feature> allFeature, int index)
{
	//allFeature  为有序链表

	list<Ml_Base_Feature> qlTestSample;
	list<Ml_Base_Feature> qlTrainSample;
	list<list<Ml_Base_Feature>> qlCLslist;

	int ClsArr[ML_CLASS_UNREV] = { 0 };
	String spilteSyl = "_";

	for (int cls = 0; cls < ML_CLASS_UNREV; cls++)
	{
		list<Ml_Base_Feature> temp;

		for (int i = 0; i < allFeature.size(); i++)
		{
			int ClsIndex = allFeature.at(i).iLabel;
			ClsArr[ClsIndex]++;
			if (ClsIndex == cls)
			{
				temp.push_back(allFeature.at(i));
			}
		}
		qlCLslist.push_back(temp);
	}


	for (int cls = 0; cls < ML_CLASS_UNREV; cls++)
	{
		list<Ml_Base_Feature> temp = qlCLslist.at(cls);
		float space = temp.size()*1.0 / 10;

		for (int i = 0; i < temp.size(); i++)
		{
			if (i >= index*space && i < (index + 1)*space)
			{
				qlTestSample.push_back(temp.at(i));
			}
			else
			{
				qlTrainSample.push_back(temp.at(i));
			}
		}
	}

	for (int i = 0; i < qlTestSample.size(); i++)
	{
		qlTrainSample.push_back(qlTestSample.at(i));
	}

	return qlTrainSample;
}

void MachineLearn::buildMlClassfierTest2(String trainPath, int iMlMode, char* testrResultPath, double param0, double param1)
{
	//建立训练样本
	int iRet = 0;
	int isize = 0;
	int iTestIndex = 0;
	list<Ml_Base_Feature> allFeature = computerFeature(trainPath, computerFeatureMode);
	isize = allFeature.size();

	if (isize == 0)
	{
		return;
	}

	for (iTestIndex = 0; iTestIndex < 10; iTestIndex++)
	{
		Mat trainData;
		Mat trainLabel;
		Mat matTemp;
		vector<int> responsees_temp;
		//random_shuffle(allFeature.begin(), allFeature.end());

		list<Ml_Base_Feature> templist = get10FoldCrossValidation(allFeature, iTestIndex);

		for (auto temp : allFeature)
		{
			responsees_temp.push_back(temp.iLabel);
			matTemp.push_back(temp.featureInfo);
		}

		Mat(responsees_temp).copyTo(trainLabel);

		if (LP_ML_REDUCED_DIM_PCA == param1)
		{
			lppca lpPcaTemp(matTemp, matTemp.rows, 0.9);
			trainData = lpPcaTemp.getFeatureReductionDimData();
		}
		else
		{
			trainData = matTemp;
		}

		Ptr<StatModel> pClsModel;
		char cParam0[20];
		char cParam1[20];
		sprintf(cParam0, "%.4lf", param0);
		sprintf(cParam1, "%.4lf", param1);
		pClsModel = buildClassifier(trainData, trainLabel, 0.9, iMlMode, param0, param1);

		//保存测试结果
		testAndSaveClassifier(pClsModel, trainData, trainLabel, templist, iMlMode, 0.9, 0, testrResultPath, cParam0, cParam1);

		responsees_temp.clear();
	}

	return;
}

void MachineLearn::buildMlClassfierTest2(list<Ml_Base_Feature> allFeature, int iMlMode, char* testrResultPath, double param0, double param1)
{
	//建立训练样本
	int iTestIndex = 0;
	int isize = allFeature.size();

	if (isize == 0)
	{
		return;
	}

	for (iTestIndex = 0; iTestIndex < 10; iTestIndex++)
	{
		Mat trainData;
		Mat trainLabel;
		Mat matTemp;
		vector<int> responsees_temp;

		list<Ml_Base_Feature> templist = get10FoldCrossValidation(allFeature, iTestIndex);
		for (int i = 0; i < isize; i++)
		{
			Ml_Base_Feature temp = templist.at(i);
			if (!temp.FeatureInfo.empty())
			{
				matTemp.push_back(temp.FeatureInfo);
				responsees_temp.push_back(temp.iLabel);
			}
		}

		Mat(responsees_temp).copyTo(trainLabel);

		if (LP_ML_REDUCED_DIM_PCA == param1)
		{
			lppca lpPcaTemp(matTemp, matTemp.rows, 0.8);
			trainData = lpPcaTemp.getFeatureReductionDimData();
		}
		else
		{
			trainData = matTemp;
		}

		Ptr<StatModel> pClsModel;
		pClsModel = buildClassifier(trainData, trainLabel, 0.9, iMlMode, param0, param1);

		char cParam0[20];
		char cParam1[20];
		sprintf(cParam0, "%.4lf", param0);
		sprintf(cParam1, "%.4lf", param1);

		//保存测试结果
		testAndSaveClassifier(pClsModel, trainData, trainLabel, templist, iMlMode, 0.9, 0, testrResultPath, cParam0, cParam1);

		responsees_temp.clear();
	}

	return;
}


int MachineLearn::buildMlClassfierAndSave(String trainPath, String mlModelSavePath, int iMlMode, double param0, double param1)
{
	if (m_bIsTrainSample)
	{
		return 0;
	}
	//建立训练样本
	Mat trainData;
	Mat trainLabel;
	buildMlClassfierSampleData(String::fromStdString(trainPath), &trainData, &trainLabel);

	//建立模型
	Ptr<StatModel> pClsModel;
	pClsModel = buildClassifier(trainData, trainLabel, 1, iMlMode, param0);

	if (pClsModel->empty())
	{
		return -1;
	}

	//保存测试结果
	pClsModel->save(mlModelSavePath.c_str());

	m_bIsTrainSample = true;

	return 0;
}

Ptr<StatModel> MachineLearn::buildMlClassfier(String trainPath, int iMlMode, double param0, double param1)
{
	//建立模型
	Ptr<StatModel> pClsModel;
	if (m_bIsTrainSample)
	{
		return pClsModel;
	}
	//建立训练样本
	Mat trainData;
	Mat trainLabel;
	buildMlClassfierSampleData(String::fromStdString(trainPath), &trainData, &trainLabel);

	//建立模型
	pClsModel = buildClassifier(trainData, trainLabel, 1, iMlMode, param0);

	return pClsModel;
}


MachineLearn::~MachineLearn()
{

}

MachineLearn::~MachineLearn()
{
}
