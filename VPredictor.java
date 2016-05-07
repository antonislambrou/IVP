import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import java.lang.reflect.Method;
import java.util.Arrays;

class VPredictor implements VPInterface{

	private int numInstances;
	private int numClasses;
	private double maxp;
	private Instances trainData;
	public double[][] matrix;
	public double low = 0;
	public double up = 0;
	
	//base classifier - must support double[] distributionForInstance();
	private Classifier classifier;
	
	//constructor with parameter
	public VPredictor(Classifier classifier)
	{
		this.classifier =  classifier; 
	}
	
	//default constructor
	public VPredictor()
	{
		this.classifier = new NaiveBayes(); //default is Bayes'
	}
	
	public void setClassifier(Classifier classifier)
	{
		this.classifier =  classifier; 
	}
	//train the base classifier
	public void buildVennPredictor(Instances trainData) throws Exception
	{
		this.trainData = trainData;
		numInstances = trainData.numInstances();
		numClasses = trainData.numClasses();
		
		classifier.buildClassifier(trainData);
	}
	
	public double classifyInstance(Instance instance) throws Exception
	{
		this.matrix = new double[numClasses][numClasses];
		double actual = instance.classValue();
		for(int i=0;i<numClasses;i++)
		{
			instance.setClassValue(i);
			this.trainData.add(instance);		
			classifier.buildClassifier(trainData);
			int[] taxonomy = taxonomize1();
			int test_category = taxonomy[numInstances];
			//System.out.printf("%d\n",test_category);
			for(int k = 0;k<numClasses;k++)
			{
				matrix[i][k] = frequency(k,test_category,taxonomy);
				//System.out.printf("%.4f ",matrix[i][k]);
			}
			this.trainData.delete(numInstances);
			//System.out.printf("\n");
		}
		//System.out.printf("\n");
		instance.setClassValue(actual);
		int predict = (int) prediction(matrix);
		double min = 2;
		double max = -1;
		
		for(int j = 0;j<numClasses;j++)
		{
			if(matrix[j][predict] > max)
			{
				max = matrix[j][predict];
			}
			if(matrix[j][predict] < min)
			{
				min = matrix[j][predict];
			}
		}
		this.low = min;
		this.up = max;
		return (double) predict;
	}
	
	public double IclassifyInstance(Instances calib_set, Instance instance) throws Exception
	{
		this.matrix = new double[numClasses][numClasses];
		double actual = instance.classValue();
		for(int i=0;i<numClasses;i++)
		{
			instance.setClassValue(i);
			calib_set.add(instance);
			int[] taxonomy = Itaxonomize1(calib_set);
			int test_category = taxonomy[calib_set.numInstances()-1];
			//System.out.printf("%d\n",test_category);
			for(int k = 0;k<numClasses;k++)
			{
				matrix[i][k] = Ifrequency(calib_set,k,test_category,taxonomy);
				//System.out.printf("%.4f ",matrix[i][k]);
			}
			calib_set.delete(calib_set.numInstances()-1);
			//System.out.printf("\n");
		}
		//System.out.printf("\n");
		instance.setClassValue(actual);
		int predict = (int) prediction(matrix);
		double min = 2;
		double max = -1;
		
		for(int j = 0;j<numClasses;j++)
		{
			if(matrix[j][predict] > max)
			{
				max = matrix[j][predict];
			}
			if(matrix[j][predict] < min)
			{
				min = matrix[j][predict];
			}
		}
		this.low = min;
		this.up = max;
		return (double) predict;
	}
	
	private double prediction(double[][] matrix)
	{
		double max = -1;
		double predict = 0;
		for(int i=0;i<numClasses;i++)
		{
			double sum = 0;
			for(int j=0;j<numClasses;j++)
			{
				sum += matrix[j][i];
			}
			sum /= numClasses;
		
			if(sum > max)
			{
				max = sum;
				predict = i;
			}
		}
		
		return predict;
	}
	public double frequency(int k,int test_category,int[] taxonomy)
	{
		int f = 0;
		int c = 0;
		int ctr = 0;
		for(double i : taxonomy)
		{
			double label = this.trainData.instance(ctr++).classValue();
			if(i == test_category)
			{		
				c += 1;
				if(label == k)
				{
					f++;
				}
				
			}
			
		}
		
		return (double)f/(double)c;
	}
	public double Ifrequency(Instances calib_set,int k,int test_category,int[] taxonomy)
	{
		int f = 0;
		int c = 0;
		int ctr = 0;
		for(double i : taxonomy)
		{
			double label = calib_set.instance(ctr++).classValue();
			if(i == test_category)
			{		
				c += 1;
				if(label == k)
				{
					f++;
				}
				
			}
			
		}
		
		return (double)f/(double)c;
	}
	//VP1
	public int[] taxonomize1() throws Exception
	{
		int[] taxonomy = new int[numInstances + 1];
		int c = 0;
		for(Instance inst : this.trainData)
		{
			taxonomy[c] = (int) classifier.classifyInstance(inst);
			c++;
		}
		
		return taxonomy;
	}
	public int[] Itaxonomize1(Instances calib_set) throws Exception
	{
		int[] taxonomy = new int[calib_set.numInstances()];
		int c = 0;
		for(Instance inst : calib_set)
		{
			taxonomy[c] = (int) classifier.classifyInstance(inst);
			c++;
		}
		
		return taxonomy;
	}
	
	//VP2
	public int[] taxonomize2() throws Exception
	{
		int[] taxonomy = new int[numInstances + 1];
		int c = 0;
		for(Instance inst : this.trainData)
		{
			double[] dist = classifier.distributionForInstance(inst);
			double max = -1;
			int index = 0;
			//find max index
			for(int i = 0;i<numClasses;i++)
			{
				if(dist[i] > max)
				{
					max = dist[i];
					index = i;
				}
			}
			taxonomy[c++] = index; 
		}
		return taxonomy;
	}
	//VP3
	public int[] taxonomize3() throws Exception
	{
		int[] taxonomy = new int[numInstances + 1];
		int c = 0;
		for(Instance inst : this.trainData)
		{
			double[] dist = classifier.distributionForInstance(inst);
			
			double max = -1;
			int index = 0;
			int index2 = 0;
			//find max index
			for(int i = 0;i<numClasses;i++)
			{
				//System.out.printf("%.4f ",dist[i]);
				if(dist[i] > max)
				{
					max = dist[i];
					index2 = index;
					index = i;
					
				}
			}
			Arrays.sort(dist);
			//System.out.printf("%.4f %.4f\n",dist[numClasses-1],dist[numClasses-2]);
			//System.out.printf("\n");
			if(dist[numClasses-1] > 0.5)
				taxonomy[c++] = index;
			else
				taxonomy[c++] = index2; 
		}
		return taxonomy;
	}
	
	public int[] taxonomize4() throws Exception
	{
		int[] taxonomy = new int[numInstances + 1];
		int c = 0;
		double array[] = new double[numInstances + 1];
		double median = 0;
		double max = -1;
		
		for(Instance inst : this.trainData)
		{
			double[] dist = classifier.distributionForInstance(inst);
			Arrays.sort(dist);
			array[c] = dist[numClasses-1];
			//System.out.printf("%.4f\n",array[c]);
			c++;
		}
		
		Arrays.sort(array);
		median = array[(numInstances) / 2];
		//System.out.printf("%.4f\n",median);
		c=0;
		
		for(Instance inst : this.trainData)
		{
			double[] dist = classifier.distributionForInstance(inst);
			int index = 0;
			max = -1;
			
			//find max index
			for(int i = 0;i<numClasses;i++)
			{
				if(dist[i] > max)
				{
					max = dist[i];
					index = i;
						
				}
			}
			//System.out.printf("%d\n",index);
			if(max >= median)
			{
				taxonomy[c++] = index;
			}
			else
			{
				taxonomy[c++] = index + numClasses;
			}
		}
		return taxonomy;
	}
	
	
		
}
