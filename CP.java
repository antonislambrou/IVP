import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.*;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import java.io.Serializable;

class CP implements Serializable{
	private static final long serialVersionUID = 1L;

	//non-conformity scores
	private double[] a;
	
	private int numInstances;
	private int numClasses;
	private double largest_Pvalue;
	private Instances trainData;
	
	//base classifier - must support distributionForInstance()
	private Classifier classifier;
	
	//constructor with parameter
	public CP(Classifier classifier)
	{
		this.classifier =  classifier; 
	}
	
	//default constructor
	public CP()
	{
		this.classifier = new NaiveBayes(); //default is Bayes'
	}
	
	public void setClassifier(Classifier classifier)
	{
		this.classifier = classifier;
	}
	public void setData(Instances trainData)
	{
		this.trainData = trainData;
	}
	//train the base classifier and generate non-conformity (alpha) scores
	public void buildConformalPredictor(Instances trainData) throws Exception
	{
		numInstances = trainData.numInstances();
		numClasses = trainData.numClasses();
		a = new double[numInstances];
		
		classifier.buildClassifier(trainData);
		
		for(int d=0;d<numInstances;d++)
		{
			Instance i = trainData.instance(d);
			a[d] = calculateNonConformityScore(i);	
		}
	}
	
	public double calculateNonConformityScore(Instance instance) throws Exception
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
	
	//get a p-value - transductive (current is already in trainData)
	public double getPvalue() throws Exception
	{
		int ctr=0;
		for(int d=0;d<numInstances;d++)
		{
			if(a[d] >= a[numInstances-1])
				ctr++;
		}
		
		return (double) ctr / (double) numInstances;

	}
	
	//get a p-value for each possible class of instance
	public double[] getPvalues(Instance instance) throws Exception
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
			
			double n = calculateNonConformityScore(instance);
			int ctr=0;
			for(int d=0;d<numInstances;d++)
			{
				if(a[d] >= n)
					ctr++;
			}
			double pvalue = (double) ctr / (double) (numInstances + 1);
			p[c] = pvalue;
			//delete the test instance.
			trainData.delete(numInstances);
		}
		//restore instance's class value
		instance.setClassValue(cl);
		
		return p;
	}
	
	//return class with highest p-value given an instance
	public double classifyInstance(Instance instance) throws Exception
	{
		double[] pvalues = getPvalues(instance);
		double pred = 0;
		double largest = 0;
		
		for(int c=0;c<numClasses;c++)
		{
			if(pvalues[c] > largest){
				largest = pvalues[c];
				pred = c;
			}	
		}
		
		this.largest_Pvalue = largest;
		return pred;
	}
	
	//return class with highest p-value given pvalues
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
		
		this.largest_Pvalue = largest;
		return pred;
	}
	
	
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
	
	//return 1-second_largest_pvalue as confidence for the classification
	//NOTE: this method must not be called before classifyInstance() !
	public double getConfidence(double[] p) throws Exception
	{
		double second_largest = 0;
		
		for(int c=0;c<numClasses;c++)
		{
			if ((p[c] != largest_Pvalue) && (p[c] > second_largest))
				second_largest = p[c];
		}
		
		return 1-second_largest;
	}
	
	
	//return 1-second_largest_pvalue as confidence for the classification
	//NOTE: this method must not be called before classifyInstance() !
	public double getConfidence(Instance instance) throws Exception
	{
		double second_largest = 0;
		double[] p = getPvalues(instance);
		
		for(int c=0;c<numClasses;c++)
		{
			if ((p[c] != largest_Pvalue) && (p[c] > second_largest))
				second_largest = p[c];
		}
		
		return 1-second_largest;
	}
	
	
	//given a confidence level return a "predictive region" (a set of classifications)
	public double[] getRegion(double confidence, Instance instance) throws Exception
	{
		double err = 1-confidence;
		int ctr = 0;
		double[] p = getPvalues(instance);
		
		for (int i=0; i<numClasses;i++)
	    	if(p[i] > err)
	    		ctr++;
	    
	    double[] region = new double[ctr];
	    ctr=0;
	    
	    for (int i=0; i<numClasses;i++)
	    	if(p[i] > err)
	    		region[ctr++] = i;
	    
	    return region;
	}
	
	//given a confidence level return a "predictive region" (a set of classifications)
	public double[] getRegion(double confidence, double[] p) throws Exception
	{
		double err = 1-confidence;
		int ctr = 0;
		
		for (int i=0; i<numClasses;i++)
	    	if(p[i] > err)
	    		ctr++;
	    
	    double[] region = new double[ctr];
	    ctr=0;
	    
	    for (int i=0; i<numClasses;i++)
	    	if(p[i] > err)
	    		region[ctr++] = i;
	    
	    return region;
	}
	public double getACC(Instances instances) throws Exception
	{
		int n,i,prediction,accuracy;
		n = instances.numInstances();
		accuracy = 0;
		for(i=0;i<n;i++)
		{
			Instance inst = instances.instance(i);
			prediction = (int)classifyInstance(inst);
			if(prediction == (int)inst.classValue())
			{
				accuracy++;
			}
		}
		return (double)accuracy/(double)n;
	}
	public double getCertainty(Instances instances, double confidence)
	{
		int n,i,certainty;
		n = instances.numInstances();
		certainty = 0;
		for(i=0;i<n;i++)
		{
			Instance inst = instances.instance(i);
			double r = getRegion(confidence,inst);
			if(r.length < 2)
			{
				certainty++;
			}
		}
		return (double)certainty/(double)n;
	}
	public double getError(Instances instances, double confidence)
	{
		int n,i,error;
		n = instances.numInstances();
		error = 0;
		for(i=0;i<n;i++)
		{
			Instance inst = instances.instance(i);
			
		}
		return error;
	}
}
