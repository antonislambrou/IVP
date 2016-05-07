import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.File;
import weka.classifiers.functions.*;

class Run
{
	public static Instances loadData(String filename) throws Exception
	{
		ArffLoader loader = new ArffLoader();
	    loader.setSource(new File(filename));
	    Instances data = loader.getDataSet();
	    data.setClassIndex(data.numAttributes() - 1);
    	System.out.printf("Data %s loaded.\n",filename);
    	return data;
	}
	
	public static void displayResults(CV cv)
	{
		double c1,c2,c3,c4;
		double e1,e2,e3,e4;
		c1 = cv.getCertainty(0.99)*100; e1 = cv.getError(0.99)*100;
		c2 = cv.getCertainty(0.95)*100; e2 = cv.getError(0.95)*100;
		c3 = cv.getCertainty(0.90)*100; e3 = cv.getError(0.90)*100;
		c4 = cv.getCertainty(0.85)*100; e4 = cv.getError(0.85)*100;
		
		System.out.printf("\n** Cross validation completed. Displaying results ** \n\n");
		System.out.printf("Accuracy: %.2f%%\n\n",cv.getAccuracy()*100);
		System.out.printf("Confidence level:     99%%\t    95%%\t\t    90%%\t\t    85%%\n");
		System.out.printf("------------------------------------");
		System.out.printf("------------------------------------\n");
		System.out.printf("Certainty       : %6.2f%%\t%6.2f%%\t\t%6.2f%%\t\t%6.2f%%\n",c1,c2,c3,c4);
		System.out.printf("Error           : %6.2f%%\t%6.2f%%\t\t%6.2f%%\t\t%6.2f%%\n\n",e1,e2,e3,e4);
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
			
			/*Neural net classifier */
			MultilayerPerceptron ann = new MultilayerPerceptron();
			ann.setValidationSetSize(10);
			CPredictor cp = new CPredictor(ann); 
		
			//create CP
			//CPredictor cp = new CPredictor(); //default is Naive Bayes CP
			CV cv = new CV(cp,folds); //create a CV experiment object.
		    //load data
			Instances data = loadData(args[0]);
    		//perform experiment on data
	    	cv.performCV(data);
	    	displayResults(cv);
    	
    	}catch(Exception e){
			e.printStackTrace();
		}
	}
}
