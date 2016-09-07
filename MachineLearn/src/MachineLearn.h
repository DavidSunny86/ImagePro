#pragma once
#include "MachineLearnUtils.h"
#include "ImageBaseUtils.h"

typedef struct ML_BASE_FEATURE
{
	Mat    featureInfo;
	int    iLabel;
	String sLable;
	uchar  ucRev[2];
}Ml_Base_Feature;



typedef enum ML_CLS_TYPE
{
	ML_CLASS_ONE,
	ML_CLASS_TWO,
	ML_CLASS_THREE,
	ML_CLASS_UNREV
};


typedef enum ML_METHOD_TYPE
{
	ML_METHOD_ADABOOST,
	ML_METHOD_RTREES,
	ML_METHOD_MLP,
	ML_METHOD_KNN,
	ML_METHOD_SVM,
	ML_METHOD_NBAYES,
	ML_METHOD_UNREV
}ML_METHOD_TYPE;


typedef enum ML_FEATURE_TYPE
{
	ML_TEXT_FEATURE,
	ML_COLOR_FEATURE,
	ML_AVE_COLOR_FEATURE,
	ML_CCV_COLOR_FEATURE,
	ML_CCV_VECTOR_FEATURE,
	ML_CCVHSV_COLOR_FEATURE,
	ML_SHAPE_FEATURE,
	ML_SHAPECONTEXT_FEATURE,
	ML_TEXT_GLCM,
	ML_TEXT_HIST,
	ML_FEATURE_MAX
};


const int class_count = CT_THREE;
const int computerFeatureMode = ML_TEXT_FEATURE;


typedef struct ML_PRE_CLS_INFO
{
	String srcClsStyle;
	int    sameStyleNum;
	int    iTrainNum;
	int    iTestNum;
	int    preTrainClsCurrentNum;
	int    preTestClsCurrentNum;

	double preTrainClsPre[ML_CLASS_UNREV];
	double preTestClsPre[ML_CLASS_UNREV];

}ML_RPE_CLS_INFO;

class MachineLearn
{
public:
	MachineLearn();
	MachineLearn(String  clsModeSavePath, int iMlMode);
	MachineLearn(String trainSamplePath, String clsModeSavePath, int iMlMode, int param0, int param1);

	Ptr<StatModel> loadMlClassify(String clsPath, int iMlMode);
	void loadMlCls(String clsPath, int iMlMode);

	int buildMlClassfierAndSave(String trainPath, String mlModelSavePath, int iMlMode, double param0, double param1);

	int predictCls(const Ptr<StatModel>& model, const Mat pendSample, int iMlModel);

	list<Ml_Base_Feature> computerFeature(String dataPath, int mode);

	Ptr<TrainData> prepareTrainData(const Mat& data, const Mat& responses, int ntrain_samples);

	int buildMlClassfierSampleData(String trainPath, Mat* data, Mat* responsees, int iReducedDimMode);

	inline TermCriteria setIterCondition(int iters, double eps);

	void buildMlClassfierTest(Mat data, Mat response, float ration, int iMlMode, char* resultPath, double params0);

	void buildMlClassfierTest(String trainPath, int iMlMode, char* testrResultPath, double param0, double param1);

	list<Ml_Base_Feature> get10FoldCrossValidation(list<Ml_Base_Feature> allFeature, int index);

	bool isLoadMlCls();

	~MachineLearn();

private:

	//随机树分类
	Ptr<StatModel> buildRtreesClassifier(Mat data, Mat  responses, int ntrain_samples, double maxDepth, double iter);
	
	Ptr<StatModel> buildAdaboostClassifier(Mat data, Mat  responses, int ntrain_samples, double param0, double param1);
	//多层感知机分类（ANN）
	Ptr<StatModel> buildMlpClassifier(Mat data, Mat  responses, int ntrain_samples, double param0, double param1);

	Ptr<StatModel> buildNbayesClassifier(Mat data, Mat  responses, int ntrain_samples);

	Ptr<StatModel> buildKnnClassifier(Mat data, Mat  responses, int ntrain_samples, int K);

	//svm分类
	Ptr<StatModel> buildSvmClassifier(Mat data, Mat  responses, int ntrain_samples);


public:

	bool m_bIsTrainSample = false;
	bool m_bIsLoadClsModel = false;
	int  m_iMlModel;
	Ptr<StatModel> m_pMlCls;
};
