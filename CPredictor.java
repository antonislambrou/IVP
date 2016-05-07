import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import java.lang.reflect.Method;

class CPredictor implements CPInterface{

	//non-conformity scores
	private double[] a;
	private int numInstances;
	private int numClasses;
	private double maxp;
	private Instances trainData;
	
	//base classifier - must support double[] distributionForInstance();
	private Classifier classifier;
	
	//constructor with parameter
	public CPredictor(Classifier classifier)
	{
		this.classifier =  classifier; 
	}
	
	//default constructor
	public CPredictor()
	{
		this.classifier = new NaiveBayes(); //default is Bayes'
	}
	
	public void setClassifier(Classifier classifier)
	{
		this.classifier =  classifier; 
	}

	//train the base classifier and generate non-conformity (alpha) scores.
	public void buildConformalPredictor(Instances trainData) throws Exception
	{
		this.trainData = trainData;
		numInstances = trainData.numInstances();
		numClasses = trainData.numClasses();
		a = new double[numInstances];
		
		classifier.buildClassifier(trainData);
		
		for(int d=0;d<numInstances;d++)
		{
			Instance i = trainData.instance(d);
			a[d] = getNonConformityOf(i);	
		}
	}
	
	public double getNonConformityOf(Instance instance) throws Exception
	{
		//sensitivity parameter
		double gamma = 1;
		//own membership of instance to class
		int own = (int) instance.classValue();
		
		double[] dist = new double[numClasses];
		dist = classifier.distributionForInstance(instance);
		
		//membership of instance to rest of the classes
		double others = 0;

		for(int q=0;q<numClasses;q++)
		{
			if(q!=own)
			{
				others+=dist[q]; 
			}
		}
		//non-conformity score
		return others-dist[own]*gamma;	//default non-conformity measure
	}
	
	//get a p-value for each possible classification of instance.
	public double[] getPvaluesOf(Instance instance) throws Exception
	{
		double[] p = new double[numClasses];
		double cl = instance.classValue();
		//try every possible class
		for(int c=0;c<numClasses;c++)
		{
			instance.setClassValue(c);
			//add the test instance
			trainData.add(instance);
			buildConformalPredictor(trainData);
			
			double n = getNonConformityOf(instance);
			int ctr=0;
			for(int d=0;d<numInstances;d++)
			{
				if(a[d] >= n)
					ctr++;
			}
			double pvalue = (double) ctr / (double) (numInstances);
			p[c] = pvalue;
			//delete the test instance.
			trainData.delete(numInstances-1);
		}
		//restore instance's class value
		instance.setClassValue(cl);
		
		return p;
	}
	
	//return classification with highest p-value, given pvalues.
	public double classifyInstance(double[] pvalues) throws Exception
	{
		double pred = 0;
		double largest = 0;
		
		for(int c=0;c<numClasses;c++)
		{
			if(pvalues[c] > largest){
				largest = pvalues[c];
				pred = c;
			}	
		}
		this.maxp = largest;
		return pred;
	}
	
	//return highest p-value.
	public double getCredibility(double[] pvalues) throws Exception
	{
		double largest = 0;
		for(int c=0;c<numClasses;c++)
		{
			if(pvalues[c] > largest){
				largest = pvalues[c];
			}	
		}	
		return largest;
	}
	
	//return 1-second_maxp as confidence for the classification.
	public double getConfidence(double[] pvalues) throws Exception
	{
		double second_largest = 0;
		classifyInstance(pvalues);
		
		for(int c=0;c<numClasses;c++)
		{
			if ((pvalues[c] != maxp) && (pvalues[c] > second_largest))
				second_largest = pvalues[c];
		}
		
		return 1-second_largest;
	}
	
	//given a confidence level return a prediction region (a set of classifications).
	public double[] getRegion(double[] pvalues, double confidence) throws Exception
	{
		double err = 1-confidence;
		int ctr = 0;
		
		//find ctr
		for (int i=0; i<numClasses;i++){
	    	if(pvalues[i] > err)
	    		ctr++;
	    }
	    double[] region = new double[ctr];
	    ctr=0;
	    
	    for (int i=0; i<numClasses;i++)
	    	if(pvalues[i] > err)
	    		region[ctr++] = i;
	    
	    return region;
	}
	
	//return 1 if the prediction of the instance, given its pvalues, is correct.
	public int getAccuracy(double[] pvalues, Instance instance) throws Exception
	{
		int acc = 0;
		int prediction = (int)classifyInstance(pvalues);
		if(prediction == (int)instance.classValue())
		{
			acc = 1;
		}
		return acc;
	}
	
	//return 1 if the region size of the confidence level is less than or equal 1. 
	public int getCertainty(double[] pvalues, double confidence) throws Exception
	{
		int certainty = 0;
		double reg[] = getRegion(pvalues,confidence);
		if(reg.length < 2)
		{
			certainty = 1;
		}
		return certainty;
	}
	
	//return 1 if the prediction region at the given confidence level does not contain
	//the classification of the given instance.
	public int getError(double[] pvalues, double confidence, Instance instance) throws Exception
	{
		int error = 1;
		int instance_class = (int)instance.classValue();
		double[] region = getRegion(pvalues,confidence);

		for(double i : region)
		{
			if((int) i == instance_class)
			{
				error = 0;
			}
		}
		
		return error;
	}
}
