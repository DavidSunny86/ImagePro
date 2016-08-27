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

	iRet = buildBtnMlClassfierAndSave(trainSamplePath, clsModeSavePath, iMlMode, param0, param1);
	if (iRet != 0)
	{
		return;
	}

	m_bIsTrainSample = true;
	return;

}


List<Ml_Base_Feature> MachineLearn::computerBtnFeature(String dataPath, int mode)
{
	List<Ml_Base_Feature> btnFeatureList;

	return btnFeatureList;
}

IplImage* MachineLearn::getImgFromBtnStyleNo(String btnStyleNo)
{
	if (btnStyleNo())
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
		sClsName = "ML_METHOD_RTREES";
		break;
	case 1:
		sClsName = "ML_METHOD_ADABOOST";
		break;
	case 2:
		sClsName = "ML_METHOD_MLP";
		break;
	case 3:
		sClsName = "ML_METHOD_KNN";
		break;
	case 4:
		sClsName = "ML_METHOD_NBAYES";
		break;
	case 5:
		sClsName = "ML_METHOD_SVM";
		break;
	default:
		break;
	}

	return sClsName;
}



int MachineLearn::buildMlClassfierBtnSampleData(String trainPath, Mat* data, Mat* responsees, int iReducedDimMode)
{
	List<Ml_Base_Feature> allBtnFeature = computerBtnFeature(trainPath, computerBtnFeatureMode);
	int iRet = 0;
	int size = allBtnFeature.size();
	vector<int> responsees_temp;
	Mat matTemp;

	if (size == 0)
	{
		return  -1;
	}

	for (int i = 0; i < size; i++)
	{
		Ml_Base_Feature temp = allBtnFeature.at(i);
		responsees_temp.push_back(temp.iBtnLabel);
		matTemp.push_back(temp.btnFeatureInfo);
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
	BTN_RPE_CLS_INFO BtnPreCls[LP_BTN_CLASS_UNREV + 1] = { 0 };

	for (i = 0; i < LP_BTN_CLASS_UNREV + 1; i++)
	{
		BtnPreCls[i] = { 0 };
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i < trainNum)
			{
				train_hr += r;
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i))<<" pre cls"<< getBtnClsReTransefer((int)best_class) << "\r\n";*/
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)best_class) << "\r\n";
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i <= trainNum)
			{
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";
				train_hr += comR;
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";*/
				test_hr += comR;
			}
		}
	}

	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;
	test_hr = (nsamples_all - trainNum) > 0 ? test_hr / (nsamples_all - trainNum) : 1.;

	outSum << qSetFieldWidth(10) << left << mlStyle.c_str() << qSetFieldWidth(10) << left << param1 <<
		qSetFieldWidth(10) << left << param2 << "trainNum" << trainNum << qSetFieldWidth(1) << " TrainClsRatio =" << train_hr <<
		qSetFieldWidth(10) << " TestClsRatio =" << test_hr << "\r\n";

	for (int btncls = 0; btncls < LP_BTN_CLASS_UNREV; btncls++)
	{

		double trainHr = BtnPreCls[btncls].iTrainNum > 0 ? (BtnPreCls[btncls].preTrainClsCurrentNum*1.0) / (BtnPreCls[btncls].iTrainNum*1.0) : 0.0f;
		double testHr = BtnPreCls[btncls].iTestNum > 0 ? (BtnPreCls[btncls].preTestClsCurrentNum*1.0) / (BtnPreCls[btncls].iTestNum*1.0) : 0.0f;

		outSum << qSetFieldWidth(10) << left << BtnPreCls[btncls].srcClsStyle << qSetFieldWidth(10) << left << "All  Num" << BtnPreCls[btncls].sameBtnStyleNum << qSetFieldWidth(10) << left << "trainNum" << BtnPreCls[btncls].iTrainNum << qSetFieldWidth(1) << " TrainClsRatio =" << trainHr <<
			qSetFieldWidth(10) << " TestClsRatio =" << testHr << "\r\n";
	}


	outSum.flush();
	sumfile.close();
}


//分类预测
void MachineLearn::testAndSaveClassifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses, List<Ml_Base_Feature> srcSampleData, int iMlModel, float ntrain_samples, int rdelta, char* resultPath, char* param1, char* param2)
{
	int i, nsamples_all = data.rows;
	int trainNum = nsamples_all*ntrain_samples;
	double train_hr = 0;
	double test_hr = 0;
	BTN_RPE_CLS_INFO BtnPreCls[LP_BTN_CLASS_UNREV + 1] = { 0 };
	BTNCLS_PREDICT_PRO btnClsPrePro[LP_BTN_CLASS_UNREV + 1];

	for (i = 0; i < LP_BTN_CLASS_UNREV + 1; i++)
	{
		BtnPreCls[i] = { 0 };
		btnClsPrePro[i] = { 0 };
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i < trainNum)
			{
				train_hr += r;
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}

				BtnPreCls[responses.at<int>(i)].preTrainClsPre[best_class]++;
				/*		outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i))<<" pre cls"<< getBtnClsReTransefer((int)best_class) << "\r\n";*/
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				BtnPreCls[responses.at<int>(i)].preTestClsPre[best_class]++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				else
				{
					String btnStyleNo = srcSampleData.at(i).sBtnLable;
					IplImage* pCurImg = getImgFromBtnStyleNo(srcSampleData.at(i).sBtnLable);
					String preCls = getBtnClsReTransefer(best_class);
					String btnImgSavePath = sSaveBasePath + "/" + btnStyleNo.toStdString() + "_" + preCls.toStdString() + ".png";
					cvSaveImage(btnImgSavePath.c_str(), pCurImg);
					cvReleaseImage(&pCurImg);

					btnClsPrePro[responses.at<int>(i)].btnSrcCls = responses.at<int>(i);
					btnClsPrePro[responses.at<int>(i)].btnPreCls[best_class]++;

				}

				//outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)best_class) << "\r\n";
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i <= trainNum)
			{
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				BtnPreCls[responses.at<int>(i)].preTrainClsPre[(int)r]++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";
				train_hr += comR;
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				BtnPreCls[responses.at<int>(i)].preTestClsPre[(int)r]++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";*/

				else
				{
					String btnStyleNo = srcSampleData.at(i).sBtnLable;
					IplImage* pCurImg = getImgFromBtnStyleNo(srcSampleData.at(i).sBtnLable);
					String preCls = getBtnClsReTransefer(r);
					String btnImgSavePath = sSaveBasePath + "/" + btnStyleNo.toStdString() + "_" + preCls.toStdString() + ".png";
					cvSaveImage(btnImgSavePath.c_str(), pCurImg);
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

	for (int btncls = 0; btncls < LP_BTN_CLASS_UNREV; btncls++)
	{

		double trainHr = BtnPreCls[btncls].iTrainNum > 0 ? (BtnPreCls[btncls].preTrainClsCurrentNum*1.0) / (BtnPreCls[btncls].iTrainNum*1.0) : 0.0f;
		double testHr = BtnPreCls[btncls].iTestNum > 0 ? (BtnPreCls[btncls].preTestClsCurrentNum*1.0) / (BtnPreCls[btncls].iTestNum*1.0) : 0.0f;

		double train0 = BtnPreCls[btncls].preTrainClsPre[0] * 1.0 / BtnPreCls[btncls].iTrainNum*1.0;
		double train1 = BtnPreCls[btncls].preTrainClsPre[1] * 1.0 / BtnPreCls[btncls].iTrainNum*1.0;
		double train2 = BtnPreCls[btncls].preTrainClsPre[2] * 1.0 / BtnPreCls[btncls].iTrainNum*1.0;

		double test0 = BtnPreCls[btncls].preTestClsPre[0] * 1.0 / BtnPreCls[btncls].iTestNum*1.0;
		double test1 = BtnPreCls[btncls].preTestClsPre[1] * 1.0 / BtnPreCls[btncls].iTestNum*1.0;
		double test2 = BtnPreCls[btncls].preTestClsPre[2] * 1.0 / BtnPreCls[btncls].iTestNum*1.0;


		clsStr << qSetFieldWidth(10) << left << BtnPreCls[btncls].srcClsStyle << qSetFieldWidth(10) << left << "All  Num" << qSetFieldWidth(5) << BtnPreCls[btncls].sameBtnStyleNum << qSetFieldWidth(10) << left << "trainNum" << qSetFieldWidth(5) << BtnPreCls[btncls].iTrainNum << qSetFieldWidth(15) << " TrainClsRatio =" <<
			qSetFieldWidth(10) << trainHr << qSetFieldWidth(15) << " TestClsRatio =" << qSetFieldWidth(10) << testHr << "Train  " <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "E=" << qSetFieldWidth(10) << train0 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "F=" << qSetFieldWidth(10) << train1 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "S=" << qSetFieldWidth(10) << train2 <<
			"Test  " <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "E=" << qSetFieldWidth(10) << test0 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "F=" << qSetFieldWidth(10) << test1 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "S=" << qSetFieldWidth(10) << test2 <<
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


int MachineLearn::predictBtnCls(const Ptr<StatModel>& model, const Mat pendSample, int iMlModel)
{
	if (model->empty())
	{
		return  -1;
	}

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


int MachineLearn::predictBtnCls(const Ptr<StatModel>& model, lppca pca, const Mat pendSample, int iMlModel)
{
	Mat pcaTemp;
	int iRet = 0;

	if (!pca.m_bIsBuildPca)
	{
		return  -1;
	}

	iRet = pca.getReductDimMat(pendSample, &pcaTemp);
	int iBtnCls = predictBtnCls(model, pendSample, iMlModel);

	if (iRet != 0 || iBtnCls < 0)
	{
		return -1;
	}
	return  iBtnCls;
}

int  MachineLearn::predictBtnClsVote(vector<BTNCLS_PREDICT_VOTE> btnPredictVote, int strictMode)
{
	int btnPreBtncls[LP_BTN_CLASS_UNREV] = { 0 };
	bool bIsVote = false;
	int btnBoostPre = LP_BTN_CLASS_UNREV;
	for (int i = 0; i < btnPredictVote.size(); i++)
	{
		int btnCls = btnPredictVote.at(i).btnPreCls;
		btnPreBtncls[btnCls]++;

		if (btnPredictVote.at(i).mlMode == LP_ML_ADABOOST)
			btnBoostPre = btnCls;
	}

	// 严格投票，只有全票通过才归为一类，否则就归为简单类
	if (strictMode)
	{
		for (int i = 0; i < LP_BTN_CLASS_UNREV; i++)
		{
			if (btnPreBtncls[i] == (LP_BTN_CLASS_UNREV - 1))
			{
				bIsVote = true;
				return i;
			}
		}

		if (bIsVote == false)
		{
			return LP_BTN_CLASS_COARSE_GRAIN;
		}
	}
	else
	{
		for (int i = 0; i < LP_BTN_CLASS_UNREV; i++)
		{
			if (btnPreBtncls[i] > 1)
			{
				bIsVote = true;
				return i;
			}
		}

		if (!bIsVote)
		{
			return  LP_BTN_CLASS_COARSE_GRAIN;
		}
	}

	return LP_BTN_CLASS_COARSE_GRAIN;
}

/*针对不同种类进行判断 */
int MachineLearn::predictBtnClsVote(vector<BTNCLS_PREDICT_VOTE> btnPredictVote, double* btnClsPrePro)
{
	int btnPreBtncls[LP_BTN_CLASS_UNREV] = { 0 };
	bool bIsVote = false;
	int btnBoostPre = LP_BTN_CLASS_UNREV;

	for (int i = 0; i < btnPredictVote.size(); i++)
	{
		int btnCls = btnPredictVote.at(i).btnPreCls;
		btnPreBtncls[btnCls]++;
	}

	for (int i = 0; i < LP_BTN_CLASS_UNREV; i++)
	{
		btnClsPrePro[i] = btnPreBtncls[i] * 1.0 / (LP_BTN_CLASS_UNREV - 1)*1.0;
	}

	return 0;
}


int MachineLearn::predictBtnCls(const Mat pendSample, int iMlModel)
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
	case LP_ML_RTREES:
	{
		Ptr<RTrees> mlClsRt = RTrees::load<RTrees>(clsPath.c_str());
		int in = mlClsRt->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsRt;
			return mlClsRt;
		}
	}

	case LP_ML_ADABOOST:
	{
		Ptr<Boost> mlClsBoost = Boost::load<Boost>(clsPath.c_str());
		int in = mlClsBoost->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsBoost;
			return mlClsBoost;
		}
	}

	case LP_ML_MLP:
	{
		Ptr<ANN_MLP> mlClsAnn = ANN_MLP::load<ANN_MLP>(clsPath.c_str());
		int in = mlClsAnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsAnn;
			return mlClsAnn;
		}
	}

	case LP_ML_NBAYES:
	{
		Ptr<NormalBayesClassifier> mlClsNbayes = NormalBayesClassifier::load<NormalBayesClassifier>(clsPath.c_str());
		int in = mlClsNbayes->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsNbayes;
			return mlClsNbayes;
		}
	}

	case LP_ML_KNN:
	{
		Ptr<KNearest> mlClsKnn = KNearest::load<KNearest>(clsPath.c_str());
		int in = mlClsKnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsKnn;
			return mlClsKnn;
		}
	}

	case LP_ML_SVM:
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
	case LP_ML_RTREES:
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

	case LP_ML_ADABOOST:
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

	case LP_ML_MLP:
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

	case LP_ML_NBAYES:
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

	case LP_ML_KNN:
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

	case LP_ML_SVM:
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
		buildMlClassfierBtnSampleData(String::fromStdString(trainPath), &trainData, &trainLabel, LP_ML_REDUCED_DIM_PCA);
		//建立模型
		buildMlClassfierTest(trainData, trainLabel, 0.8, iMlMode, testrResultPath, param0);
	}

	return;
}

List<Ml_Base_Feature> MachineLearn::get10FoldCrossValidation(List<Ml_Base_Feature> allBtnFeature, int index)
{
	//allBtnFeature  为有序链表

	List<Ml_Base_Feature> qlTestSample;
	List<Ml_Base_Feature> qlTrainSample;
	List<List<Ml_Base_Feature>> qlBtnCLsList;

	int btnClsArr[LP_BTN_CLASS_UNREV] = { 0 };
	String spilteSyl = "_";

	for (int cls = 0; cls < LP_BTN_CLASS_UNREV; cls++)
	{
		List<Ml_Base_Feature> temp;

		for (int i = 0; i < allBtnFeature.size(); i++)
		{
			int btnClsIndex = allBtnFeature.at(i).iBtnLabel;
			btnClsArr[btnClsIndex]++;
			if (btnClsIndex == cls)
			{
				temp.push_back(allBtnFeature.at(i));
			}
		}
		qlBtnCLsList.push_back(temp);
	}


	for (int cls = 0; cls < LP_BTN_CLASS_UNREV; cls++)
	{
		List<Ml_Base_Feature> temp = qlBtnCLsList.at(cls);
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
	List<Ml_Base_Feature> allBtnFeature = computerBtnFeature(String::fromStdString(trainPath), computerBtnFeatureMode);
	isize = allBtnFeature.size();

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
		//random_shuffle(allBtnFeature.begin(), allBtnFeature.end());

		List<Ml_Base_Feature> tempList = get10FoldCrossValidation(allBtnFeature, iTestIndex);
		for (int i = 0; i < isize; i++)
		{
			Ml_Base_Feature temp = tempList.at(i);
			responsees_temp.push_back(temp.iBtnLabel);
			matTemp.push_back(temp.btnFeatureInfo);
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
		testAndSaveClassifier(pClsModel, trainData, trainLabel, tempList, iMlMode, 0.9, 0, testrResultPath, cParam0, cParam1);

		responsees_temp.clear();
	}

	return;
}

void MachineLearn::buildMlClassfierTest2(List<Ml_Base_Feature> allBtnFeature, int iMlMode, char* testrResultPath, double param0, double param1)
{
	//建立训练样本
	int iTestIndex = 0;
	int isize = allBtnFeature.size();

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

		List<Ml_Base_Feature> tempList = get10FoldCrossValidation(allBtnFeature, iTestIndex);
		for (int i = 0; i < isize; i++)
		{
			Ml_Base_Feature temp = tempList.at(i);
			if (!temp.btnFeatureInfo.empty())
			{
				matTemp.push_back(temp.btnFeatureInfo);
				responsees_temp.push_back(temp.iBtnLabel);
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
		testAndSaveClassifier(pClsModel, trainData, trainLabel, tempList, iMlMode, 0.9, 0, testrResultPath, cParam0, cParam1);

		responsees_temp.clear();
	}

	return;
}




int MachineLearn::buildBtnMlClassfierAndSave(String trainPath, String mlModelSavePath, int iMlMode, double param0, double param1)
{
	if (m_bIsTrainSample)
	{
		return 0;
	}
	//建立训练样本
	Mat trainData;
	Mat trainLabel;
	buildMlClassfierBtnSampleData(String::fromStdString(trainPath), &trainData, &trainLabel);

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

Ptr<StatModel> MachineLearn::buildBtnMlClassfier(String trainPath, int iMlMode, double param0, double param1)
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
	buildMlClassfierBtnSampleData(String::fromStdString(trainPath), &trainData, &trainLabel);

	//建立模型
	pClsModel = buildClassifier(trainData, trainLabel, 1, iMlMode, param0);

	return pClsModel;
}


MachineLearn::~MachineLearn()
{

}
#include "MachineLearn.h"
#include "lppca.h"

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

	iRet = buildBtnMlClassfierAndSave(trainSamplePath, clsModeSavePath, iMlMode, param0, param1);
	if (iRet != 0)
	{
		return;
	}

	m_bIsTrainSample = true;
	return;

}


List<Ml_Base_Feature> MachineLearn::computerBtnFeature(String dataPath, int mode)
{
	List<Ml_Base_Feature> btnFeatureList;
#if 0
	cv::Mat srcFeatureMat;
	String splitSym = "-";

	if (dataPath.isEmpty())
	{
		return  btnFeatureList;
	}

	/*获取数据当前数据库中条目中的u_id*/
	LpObjStore* store = LpObjStore::getInstance();
	StringList u_idlist = store->fetchObj(0, 10000);

	if (u_idlist.size() == 0)
	{
		return btnFeatureList;
	}


	foreach(String id, u_idlist)
	{
		//step1 
		LpButton * pCurBtn = store->loadObj(id);
		if (pCurBtn == NULL)
		{
			YDEBUG() << "load  Btn is error !!" << endl;
			continue;
		}
		cv::Mat curFeatureMat;
		String qsFileName = pCurBtn->m_fileName;
		String curStyleNo = pCurBtn->m_style_no;
		YDEBUG() << "style" << curStyleNo << "lpfName" << qsFileName << endl;
		store->freeObj(pCurBtn);

		//step2::
		String qsCurFullPath = dataPath + "/" + qsFileName + ".lpf";
		IplImage *curImg;
		IplImage *pCurMaskImg;
		CLPB curlLPB;
		LPDATA objLPB;
		curlLPB.m_cstrSrcRoot = qsCurFullPath;
		QFile curlpfFile(qsCurFullPath);
		Ml_Base_Feature curBaseInfo;

		YDEBUG() << "qsCurFullPath == " << qsCurFullPath << endl;
		if (curStyleNo.isEmpty() || qsFileName.isEmpty())
		{
			YDEBUG() << "curstyleNo is NULL" << "lpfName is NULL" << endl;
			continue;
		}
		if (!curlpfFile.open(QIODevice::ReadOnly))
		{
			continue;
		}

		curlLPB.ReadHeader(&curlpfFile, &objLPB);


		//step3;
		/*获取当前数据库条码中的前景图像与mask图像	*/
		int curImgIdx = CLPB::getImgIndex("front.jpg");
		curImg = loadImageFromLpf(curlpfFile, &objLPB, curImgIdx);
		curImgIdx = CLPB::getImgIndex("frontmask.jpg");
		pCurMaskImg = loadImageFromLpf(curlpfFile, &objLPB, curImgIdx);
		curlpfFile.close();

		if (curImg == NULL || curImg == nullptr || pCurMaskImg == NULL || pCurMaskImg == nullptr)
		{
			YDEBUG() << "curImg is NULL" << " || pCurMaskImg is NULL" << endl;
			continue;
		}
		YDEBUG() << " step3 get Img  is OK" << endl;

		//shape mat computer
		if (mode == LP_SHAPE_FEATURE)
		{
			//computerImgShape(pCurMaskImg, curFeatureMat);
		}
		else if (mode == LP_COLOR_FEATURE)
		{
			computeColorFeature(curImg, pCurMaskImg, curFeatureMat);
		}
		else if (mode == LP_TEXT_FEATURE)
		{
			cv::Mat  textFeatureROI = computeTextureImgROI(curImg, pCurMaskImg);
			if (textFeatureROI.empty())
			{
				YDEBUG() << curStyleNo << " computer Feature error" << endl;
				continue;
			}
			IplImage rotateTextFeatureROI(textFeatureROI);
			textFeatureROI = getRotatedImage(&rotateTextFeatureROI, 1);
			computeTextureFeature(textFeatureROI, curFeatureMat);
		}
		else if (mode == LP_TEXT_GLCM)
		{
			cv::Mat  textFeatureROI = computeTextureImgROI(curImg, pCurMaskImg);
			lpglcmtext lpglcmtext;
			if (textFeatureROI.empty())
			{
				YDEBUG() << curStyleNo << " computer Feature error" << endl;
				continue;
			}

			curFeatureMat = lpglcmtext.getSampleGlcmVectorList(textFeatureROI);
		}
		else if (mode == LP_TEXT_HIST)
		{
			IplImage *ImgGray = cvCreateImage(cvGetSize(curImg), 8, 1);
			IplImage* imgMaskNew = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), 8, 1);
			cvZero(imgMaskNew);
			CDetect detect;
			bool flag = detect.GetButtonMask(pCurMaskImg, imgMaskNew);
			if (!flag)
			{
				cvReleaseImage(&imgMaskNew);
				continue;
			}

			cvCvtColor(curImg, ImgGray, CV_BGR2GRAY);
			bool uniform = true; bool accumulate = false;
			float gray_range[] = { 0, 255 };
			const float* gray_histRange = { gray_range };
			Mat gray = cvarrToMat(ImgGray, 1);
			int gray_bins = 256;
			Mat mask_base = cvarrToMat(imgMaskNew);
			Mat gray_hist;
			calcHist(&gray, 1, 0, mask_base, gray_hist, 1, &gray_bins, &gray_histRange, uniform, accumulate);
			curFeatureMat = gray_hist.t();

			cvReleaseImage(&ImgGray);
			cvReleaseImage(&imgMaskNew);
		}

		YDEBUG() << " step4 computer text feature  mat is OK !" << endl;

		String btnLabel = curStyleNo.mid(0, curStyleNo.indexOf(splitSym));
		int btnStyle = getBtnClsTransefer(btnLabel);
		curBaseInfo.iBtnLabel = btnStyle;
		curBaseInfo.sBtnLable = curStyleNo;
		curFeatureMat.copyTo(curBaseInfo.btnFeatureInfo);
		btnFeatureList.append(curBaseInfo);

		lpReleaseImage(curImg);
		lpReleaseImage(pCurMaskImg);
	}
#endif
	return btnFeatureList;
}

IplImage* MachineLearn::getImgFromBtnStyleNo(String btnStyleNo)
{
	if (btnStyleNo.isEmpty())
	{
		return NULL;
	}
#if 0
	String dataPath = qApp->applicationDirPath() + "/Data/ButtonLib";

	LpObjStore* store = LpObjStore::getInstance();
	LpButton * pCurBtn = store->loadObj(btnStyleNo);
	if (pCurBtn == NULL)
	{
		YDEBUG() << "load  Btn is error !!" << endl;
		return NULL;
	}
	cv::Mat curFeatureMat;
	String qsFileName = pCurBtn->m_fileName;
	String curStyleNo = pCurBtn->m_style_no;
	YDEBUG() << "style" << curStyleNo << "lpfName" << qsFileName << endl;
	store->freeObj(pCurBtn);

	//step2::
	String qsCurFullPath = dataPath + "/" + qsFileName + ".lpf";
	IplImage *curImg;
	CLPB curlLPB;
	LPDATA objLPB;
	curlLPB.m_cstrSrcRoot = qsCurFullPath;
	QFile curlpfFile(qsCurFullPath);
	Ml_Base_Feature curBaseInfo;

	YDEBUG() << "qsCurFullPath == " << qsCurFullPath << endl;
	if (curStyleNo.isEmpty() || qsFileName.isEmpty())
	{
		YDEBUG() << "curstyleNo is NULL" << "lpfName is NULL" << endl;
		return NULL;
	}
	if (!curlpfFile.open(QIODevice::ReadOnly))
	{
		return NULL;;
	}

	curlLPB.ReadHeader(&curlpfFile, &objLPB);

	//step3;
	/*获取当前数据库条码中的前景图像与mask图像	*/
	int curImgIdx = CLPB::getImgIndex("front.jpg");
	curImg = loadImageFromLpf(curlpfFile, &objLPB, curImgIdx);
	curlpfFile.close();

	return curImg;
#endif


}


String MachineLearn::getClsName(int iMlModel)
{
	String sClsName;
	switch (iMlModel)
	{
	case 0:
		sClsName = "LP_ML_RTREES";
		break;
	case 1:
		sClsName = "LP_ML_ADABOOST";
		break;
	case 2:
		sClsName = "LP_ML_MLP";
		break;
	case 3:
		sClsName = "LP_ML_KNN";
		break;
	case 4:
		sClsName = "LP_ML_NBAYES";
		break;
	case 5:
		sClsName = "LP_ML_SVM";
		break;
	default:
		break;
	}

	return sClsName;
}

int MachineLearn::getBtnClsTransefer(String btnlabel)
{
	if (0 == btnlabel.compare("E"))
	{
		return 0;
	}
	if (0 == btnlabel.compare("F"))
	{
		return 1;
	}
	if (0 == btnlabel.compare("S"))
	{
		return 2;
	}
	if (0 == btnlabel.compare("SW"))
	{
		return 3;
	}
	else
	{
		return -1;
	}
}

String MachineLearn::getBtnClsReTransefer(int iBtnCls)
{
	String qsBtnCls;
	switch (iBtnCls)
	{
	case 0:
		qsBtnCls = "E";
		break;
	case 1:
		qsBtnCls = "F";
		break;
	case 2:
		qsBtnCls = "S";
		break;
	case 3:
		qsBtnCls = "SW";
		break;
	default:
		break;
	}

	return qsBtnCls;
}



int MachineLearn::buildMlClassfierBtnSampleData(String trainPath, Mat* data, Mat* responsees, int iReducedDimMode)
{
	List<Ml_Base_Feature> allBtnFeature = computerBtnFeature(trainPath, computerBtnFeatureMode);
	int iRet = 0;
	int size = allBtnFeature.size();
	vector<int> responsees_temp;
	Mat matTemp;

	if (size == 0)
	{
		return  -1;
	}

	for (int i = 0; i < size; i++)
	{
		Ml_Base_Feature temp = allBtnFeature.at(i);
		responsees_temp.push_back(temp.iBtnLabel);
		matTemp.push_back(temp.btnFeatureInfo);
	}

	if (LP_ML_REDUCED_DIM_PCA == iReducedDimMode)
	{
		lppca lpPcaTemp(matTemp, size, 0.8);
		matTemp = lpPcaTemp.getFeatureReductionDimData();
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
	BTN_RPE_CLS_INFO BtnPreCls[LP_BTN_CLASS_UNREV + 1] = { 0 };

	for (i = 0; i < LP_BTN_CLASS_UNREV + 1; i++)
	{
		BtnPreCls[i] = { 0 };
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i < trainNum)
			{
				train_hr += r;
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i))<<" pre cls"<< getBtnClsReTransefer((int)best_class) << "\r\n";*/
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)best_class) << "\r\n";
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i <= trainNum)
			{
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";
				train_hr += comR;
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";*/
				test_hr += comR;
			}
		}
	}

	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;
	test_hr = (nsamples_all - trainNum) > 0 ? test_hr / (nsamples_all - trainNum) : 1.;

	outSum << qSetFieldWidth(10) << left << mlStyle.c_str() << qSetFieldWidth(10) << left << param1 <<
		qSetFieldWidth(10) << left << param2 << "trainNum" << trainNum << qSetFieldWidth(1) << " TrainClsRatio =" << train_hr <<
		qSetFieldWidth(10) << " TestClsRatio =" << test_hr << "\r\n";

	for (int btncls = 0; btncls < LP_BTN_CLASS_UNREV; btncls++)
	{

		double trainHr = BtnPreCls[btncls].iTrainNum > 0 ? (BtnPreCls[btncls].preTrainClsCurrentNum*1.0) / (BtnPreCls[btncls].iTrainNum*1.0) : 0.0f;
		double testHr = BtnPreCls[btncls].iTestNum > 0 ? (BtnPreCls[btncls].preTestClsCurrentNum*1.0) / (BtnPreCls[btncls].iTestNum*1.0) : 0.0f;

		outSum << qSetFieldWidth(10) << left << BtnPreCls[btncls].srcClsStyle << qSetFieldWidth(10) << left << "All  Num" << BtnPreCls[btncls].sameBtnStyleNum << qSetFieldWidth(10) << left << "trainNum" << BtnPreCls[btncls].iTrainNum << qSetFieldWidth(1) << " TrainClsRatio =" << trainHr <<
			qSetFieldWidth(10) << " TestClsRatio =" << testHr << "\r\n";
	}


	outSum.flush();
	sumfile.close();
}


//分类预测
void MachineLearn::testAndSaveClassifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses, List<Ml_Base_Feature> srcSampleData, int iMlModel, float ntrain_samples, int rdelta, char* resultPath, char* param1, char* param2)
{
	int i, nsamples_all = data.rows;
	int trainNum = nsamples_all*ntrain_samples;
	double train_hr = 0;
	double test_hr = 0;
	BTN_RPE_CLS_INFO BtnPreCls[LP_BTN_CLASS_UNREV + 1] = { 0 };
	BTNCLS_PREDICT_PRO btnClsPrePro[LP_BTN_CLASS_UNREV + 1];

	for (i = 0; i < LP_BTN_CLASS_UNREV + 1; i++)
	{
		BtnPreCls[i] = { 0 };
		btnClsPrePro[i] = { 0 };
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i < trainNum)
			{
				train_hr += r;
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}

				BtnPreCls[responses.at<int>(i)].preTrainClsPre[best_class]++;
				/*		outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i))<<" pre cls"<< getBtnClsReTransefer((int)best_class) << "\r\n";*/
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				BtnPreCls[responses.at<int>(i)].preTestClsPre[best_class]++;
				if (r > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				else
				{
					String btnStyleNo = srcSampleData.at(i).sBtnLable;
					IplImage* pCurImg = getImgFromBtnStyleNo(srcSampleData.at(i).sBtnLable);
					String preCls = getBtnClsReTransefer(best_class);
					String btnImgSavePath = sSaveBasePath + "/" + btnStyleNo.toStdString() + "_" + preCls.toStdString() + ".png";
					cvSaveImage(btnImgSavePath.c_str(), pCurImg);
					cvReleaseImage(&pCurImg);

					btnClsPrePro[responses.at<int>(i)].btnSrcCls = responses.at<int>(i);
					btnClsPrePro[responses.at<int>(i)].btnPreCls[best_class]++;

				}

				//outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)best_class) << "\r\n";
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

			BtnPreCls[responses.at<int>(i)].srcClsStyle = getBtnClsReTransefer(responses.at<int>(i));
			BtnPreCls[responses.at<int>(i)].sameBtnStyleNum++;

			//if (i >= trainNum && iFirstFlag == 0)
			//{
			//	iFirstFlag++;
			//	outSum << qSetFieldWidth(10) << left << " Test Sample Begin!!" << "\r\n";
			//}

			if (i <= trainNum)
			{
				BtnPreCls[responses.at<int>(i)].iTrainNum++;
				BtnPreCls[responses.at<int>(i)].preTrainClsPre[(int)r]++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTrainClsCurrentNum++;
				}
				//outSum << qSetFieldWidth(10) << left << "train-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";
				train_hr += comR;
			}
			else
			{
				BtnPreCls[responses.at<int>(i)].iTestNum++;
				BtnPreCls[responses.at<int>(i)].preTestClsPre[(int)r]++;
				if (comR > 0.5)
				{
					BtnPreCls[responses.at<int>(i)].preTestClsCurrentNum++;
				}
				/*		outSum << qSetFieldWidth(10) << left << "test-" << "src cls" << getBtnClsReTransefer(responses.at<int>(i)) << " pre cls" << getBtnClsReTransefer((int)r) << "\r\n";*/

				else
				{
					String btnStyleNo = srcSampleData.at(i).sBtnLable;
					IplImage* pCurImg = getImgFromBtnStyleNo(srcSampleData.at(i).sBtnLable);
					String preCls = getBtnClsReTransefer(r);
					String btnImgSavePath = sSaveBasePath + "/" + btnStyleNo.toStdString() + "_" + preCls.toStdString() + ".png";
					cvSaveImage(btnImgSavePath.c_str(), pCurImg);
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

	for (int btncls = 0; btncls < LP_BTN_CLASS_UNREV; btncls++)
	{

		double trainHr = BtnPreCls[btncls].iTrainNum > 0 ? (BtnPreCls[btncls].preTrainClsCurrentNum*1.0) / (BtnPreCls[btncls].iTrainNum*1.0) : 0.0f;
		double testHr = BtnPreCls[btncls].iTestNum > 0 ? (BtnPreCls[btncls].preTestClsCurrentNum*1.0) / (BtnPreCls[btncls].iTestNum*1.0) : 0.0f;

		double train0 = BtnPreCls[btncls].preTrainClsPre[0] * 1.0 / BtnPreCls[btncls].iTrainNum*1.0;
		double train1 = BtnPreCls[btncls].preTrainClsPre[1] * 1.0 / BtnPreCls[btncls].iTrainNum*1.0;
		double train2 = BtnPreCls[btncls].preTrainClsPre[2] * 1.0 / BtnPreCls[btncls].iTrainNum*1.0;

		double test0 = BtnPreCls[btncls].preTestClsPre[0] * 1.0 / BtnPreCls[btncls].iTestNum*1.0;
		double test1 = BtnPreCls[btncls].preTestClsPre[1] * 1.0 / BtnPreCls[btncls].iTestNum*1.0;
		double test2 = BtnPreCls[btncls].preTestClsPre[2] * 1.0 / BtnPreCls[btncls].iTestNum*1.0;


		clsStr << qSetFieldWidth(10) << left << BtnPreCls[btncls].srcClsStyle << qSetFieldWidth(10) << left << "All  Num" << qSetFieldWidth(5) << BtnPreCls[btncls].sameBtnStyleNum << qSetFieldWidth(10) << left << "trainNum" << qSetFieldWidth(5) << BtnPreCls[btncls].iTrainNum << qSetFieldWidth(15) << " TrainClsRatio =" <<
			qSetFieldWidth(10) << trainHr << qSetFieldWidth(15) << " TestClsRatio =" << qSetFieldWidth(10) << testHr << "Train  " <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "E=" << qSetFieldWidth(10) << train0 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "F=" << qSetFieldWidth(10) << train1 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "S=" << qSetFieldWidth(10) << train2 <<
			"Test  " <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "E=" << qSetFieldWidth(10) << test0 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "F=" << qSetFieldWidth(10) << test1 <<
			qSetFieldWidth(1) << BtnPreCls[btncls].srcClsStyle << "-" << "S=" << qSetFieldWidth(10) << test2 <<
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


int MachineLearn::predictBtnCls(const Ptr<StatModel>& model, const Mat pendSample, int iMlModel)
{
	if (model->empty())
	{
		return  -1;
	}

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


int MachineLearn::predictBtnCls(const Ptr<StatModel>& model, lppca pca, const Mat pendSample, int iMlModel)
{
	Mat pcaTemp;
	int iRet = 0;

	if (!pca.m_bIsBuildPca)
	{
		return  -1;
	}

	iRet = pca.getReductDimMat(pendSample, &pcaTemp);
	int iBtnCls = predictBtnCls(model, pendSample, iMlModel);

	if (iRet != 0 || iBtnCls < 0)
	{
		return -1;
	}
	return  iBtnCls;
}

int  MachineLearn::predictBtnClsVote(vector<BTNCLS_PREDICT_VOTE> btnPredictVote, int strictMode)
{
	int btnPreBtncls[LP_BTN_CLASS_UNREV] = { 0 };
	bool bIsVote = false;
	int btnBoostPre = LP_BTN_CLASS_UNREV;
	for (int i = 0; i < btnPredictVote.size(); i++)
	{
		int btnCls = btnPredictVote.at(i).btnPreCls;
		btnPreBtncls[btnCls]++;

		if (btnPredictVote.at(i).mlMode == LP_ML_ADABOOST)
			btnBoostPre = btnCls;
	}

	// 严格投票，只有全票通过才归为一类，否则就归为简单类
	if (strictMode)
	{
		for (int i = 0; i < LP_BTN_CLASS_UNREV; i++)
		{
			if (btnPreBtncls[i] == (LP_BTN_CLASS_UNREV - 1))
			{
				bIsVote = true;
				return i;
			}
		}

		if (bIsVote == false)
		{
			return LP_BTN_CLASS_COARSE_GRAIN;
		}
	}
	else
	{
		for (int i = 0; i < LP_BTN_CLASS_UNREV; i++)
		{
			if (btnPreBtncls[i] > 1)
			{
				bIsVote = true;
				return i;
			}
		}

		if (!bIsVote)
		{
			return  LP_BTN_CLASS_COARSE_GRAIN;
		}
	}

	return LP_BTN_CLASS_COARSE_GRAIN;
}

/*针对不同种类进行判断 */
int MachineLearn::predictBtnClsVote(vector<BTNCLS_PREDICT_VOTE> btnPredictVote, double* btnClsPrePro)
{
	int btnPreBtncls[LP_BTN_CLASS_UNREV] = { 0 };
	bool bIsVote = false;
	int btnBoostPre = LP_BTN_CLASS_UNREV;

	for (int i = 0; i < btnPredictVote.size(); i++)
	{
		int btnCls = btnPredictVote.at(i).btnPreCls;
		btnPreBtncls[btnCls]++;
	}

	for (int i = 0; i < LP_BTN_CLASS_UNREV; i++)
	{
		btnClsPrePro[i] = btnPreBtncls[i] * 1.0 / (LP_BTN_CLASS_UNREV - 1)*1.0;
	}

	return 0;
}


int MachineLearn::predictBtnCls(const Mat pendSample, int iMlModel)
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
	case LP_ML_RTREES:
	{
		Ptr<RTrees> mlClsRt = RTrees::load<RTrees>(clsPath.c_str());
		int in = mlClsRt->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsRt;
			return mlClsRt;
		}
	}

	case LP_ML_ADABOOST:
	{
		Ptr<Boost> mlClsBoost = Boost::load<Boost>(clsPath.c_str());
		int in = mlClsBoost->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsBoost;
			return mlClsBoost;
		}
	}

	case LP_ML_MLP:
	{
		Ptr<ANN_MLP> mlClsAnn = ANN_MLP::load<ANN_MLP>(clsPath.c_str());
		int in = mlClsAnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsAnn;
			return mlClsAnn;
		}
	}

	case LP_ML_NBAYES:
	{
		Ptr<NormalBayesClassifier> mlClsNbayes = NormalBayesClassifier::load<NormalBayesClassifier>(clsPath.c_str());
		int in = mlClsNbayes->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsNbayes;
			return mlClsNbayes;
		}
	}

	case LP_ML_KNN:
	{
		Ptr<KNearest> mlClsKnn = KNearest::load<KNearest>(clsPath.c_str());
		int in = mlClsKnn->getVarCount();
		if (in)
		{
			m_pMlCls = mlClsKnn;
			return mlClsKnn;
		}
	}

	case LP_ML_SVM:
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
	case LP_ML_RTREES:
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

	case LP_ML_ADABOOST:
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

	case LP_ML_MLP:
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

	case LP_ML_NBAYES:
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

	case LP_ML_KNN:
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

	case LP_ML_SVM:
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
		buildMlClassfierBtnSampleData(String::fromStdString(trainPath), &trainData, &trainLabel, LP_ML_REDUCED_DIM_PCA);
		//建立模型
		buildMlClassfierTest(trainData, trainLabel, 0.8, iMlMode, testrResultPath, param0);
	}

	return;
}

List<Ml_Base_Feature> MachineLearn::get10FoldCrossValidation(List<Ml_Base_Feature> allBtnFeature, int index)
{
	//allBtnFeature  为有序链表

	List<Ml_Base_Feature> qlTestSample;
	List<Ml_Base_Feature> qlTrainSample;
	List<List<Ml_Base_Feature>> qlBtnCLsList;

	int btnClsArr[LP_BTN_CLASS_UNREV] = { 0 };
	String spilteSyl = "_";

	for (int cls = 0; cls < LP_BTN_CLASS_UNREV; cls++)
	{
		List<Ml_Base_Feature> temp;

		for (int i = 0; i < allBtnFeature.size(); i++)
		{
			int btnClsIndex = allBtnFeature.at(i).iBtnLabel;
			btnClsArr[btnClsIndex]++;
			if (btnClsIndex == cls)
			{
				temp.push_back(allBtnFeature.at(i));
			}
		}
		qlBtnCLsList.push_back(temp);
	}


	for (int cls = 0; cls < LP_BTN_CLASS_UNREV; cls++)
	{
		List<Ml_Base_Feature> temp = qlBtnCLsList.at(cls);
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
	List<Ml_Base_Feature> allBtnFeature = computerBtnFeature(String::fromStdString(trainPath), computerBtnFeatureMode);
	isize = allBtnFeature.size();

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
		//random_shuffle(allBtnFeature.begin(), allBtnFeature.end());

		List<Ml_Base_Feature> tempList = get10FoldCrossValidation(allBtnFeature, iTestIndex);
		for (int i = 0; i < isize; i++)
		{
			Ml_Base_Feature temp = tempList.at(i);
			responsees_temp.push_back(temp.iBtnLabel);
			matTemp.push_back(temp.btnFeatureInfo);
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
		testAndSaveClassifier(pClsModel, trainData, trainLabel, tempList, iMlMode, 0.9, 0, testrResultPath, cParam0, cParam1);

		responsees_temp.clear();
	}

	return;
}

void MachineLearn::buildMlClassfierTest2(List<Ml_Base_Feature> allBtnFeature, int iMlMode, char* testrResultPath, double param0, double param1)
{
	//建立训练样本
	int iTestIndex = 0;
	int isize = allBtnFeature.size();

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

		List<Ml_Base_Feature> tempList = get10FoldCrossValidation(allBtnFeature, iTestIndex);
		for (int i = 0; i < isize; i++)
		{
			Ml_Base_Feature temp = tempList.at(i);
			if (!temp.btnFeatureInfo.empty())
			{
				matTemp.push_back(temp.btnFeatureInfo);
				responsees_temp.push_back(temp.iBtnLabel);
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
		testAndSaveClassifier(pClsModel, trainData, trainLabel, tempList, iMlMode, 0.9, 0, testrResultPath, cParam0, cParam1);

		responsees_temp.clear();
	}

	return;
}




int MachineLearn::buildBtnMlClassfierAndSave(String trainPath, String mlModelSavePath, int iMlMode, double param0, double param1)
{
	if (m_bIsTrainSample)
	{
		return 0;
	}
	//建立训练样本
	Mat trainData;
	Mat trainLabel;
	buildMlClassfierBtnSampleData(String::fromStdString(trainPath), &trainData, &trainLabel);

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

Ptr<StatModel> MachineLearn::buildBtnMlClassfier(String trainPath, int iMlMode, double param0, double param1)
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
	buildMlClassfierBtnSampleData(String::fromStdString(trainPath), &trainData, &trainLabel);

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
