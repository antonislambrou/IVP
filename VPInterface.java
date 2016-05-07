import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

/* Venn Predictor Interface */
interface VPInterface
{	
	public void setClassifier(Classifier classifier);
	public double classifyInstance(Instance instance) throws Exception;
	public double IclassifyInstance(Instances calib_set,Instance instance) throws Exception;
	public void buildVennPredictor(Instances trainData) throws Exception;
	
}
