import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.SelectedTag;
import java.io.File;
import weka.classifiers.functions.*;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.lazy.*;
import weka.classifiers.trees.*;
import weka.classifiers.meta.MultiBoostAB;
import java.util.Random;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.AttributeSelectedClassifier;
class Run
{
	public static Instances loadData(String filename) throws Exception
	{
		ArffLoader loader = new ArffLoader();
	    loader.setSource(new File(filename));
	    Instances data = loader.getDataSet();
	    data.setClassIndex(data.numAttributes() - 1);
    	//System.out.printf("Data %s loaded.\n",filename);
    	return data;
	}
	
	
	
	public static void main(String[] args)
	{
		try{
			double accuracy;
			int folds;
			
			//check parameters (input file and number of folds)
			if(args.length < 1)
			{
				System.out.printf("Input arff file must be specified\n");
				System.exit(-1);
			}
			if(args.length > 1)
			{
				folds = Integer.parseInt(args[1]);
			}
			else //if second argument not specified folds is 10 by default.
				folds = 10;
			
			/*AdaBoostM1 */
			AdaBoostM1 ada = new AdaBoostM1();
			String[] options = {"-I", "20"};
			ada.setOptions(options);
			
			IBk knn = new IBk();
			knn.setKNN(10);
			
			J48 tree = new J48();
			
			/*Neural net classifier */
			MultilayerPerceptron ann = new MultilayerPerceptron();
			ann.setValidationSetSize(10);
			
			/* SMO */
			AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
			SMO smo = new SMO();
			RBFKernel rbf = new RBFKernel();
			rbf.setGamma(1);
			smo.setKernel(rbf);
			String[] options2 = {"-N","2"};
			smo.setOptions(options2);
			asc.setClassifier(ann);
			VPredictor vp = new VPredictor();

			CV cv = new CV(vp,folds); //create a cross validation experiment.
		    //load data
			Instances data = loadData(args[0]);
    		Random rnd = data.getRandomNumberGenerator(1);
    		data.randomize(rnd);
    		/*
    		int numInstances2 = data.numInstances();
    		int numInstances = numInstances2;
    		for(int j=0;j<numInstances2-300;j++)
    		{
    			data.remove(numInstances-1);
    			numInstances = data.numInstances();
    		}

    		System.out.printf("%d\n",data.numInstances());
    		*/
    		//perform experiment on data
	    	cv.performCV(data);
	    	//displayResults(cv);
    	
    	}catch(Exception e){
			e.printStackTrace();
		}
	}
}
