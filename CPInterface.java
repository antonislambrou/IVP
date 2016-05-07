import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

/* Conformal Predictor Interface */
interface CPInterface
{	
	public void setClassifier(Classifier classifier);
	public void buildConformalPredictor(Instances trainData) throws Exception;
	public double getNonConformityOf(Instance instance) throws Exception;
	public double[] getPvaluesOf(Instance instance) throws Exception;
	public double classifyInstance(double[] pvalues) throws Exception;
	public double getCredibility(double[] pvalues) throws Exception;
	public double getConfidence(double[] pvalues) throws Exception;
	public double[] getRegion(double[] pvalues,double confidence) throws Exception;
	public int getAccuracy(double[] pvalues,Instance instance) throws Exception;
	public int getCertainty(double[] pvalues,double confidence) throws Exception;
	public int getError(double[] pvalues,double confidence,Instance instance) throws Exception;
}
