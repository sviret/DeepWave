from DGWS import gendata as gd
from DGWS import trainCNN as tr
from DGWS import useResults as ur
import os

def testTrainer():
	##Generateurs des Datasets
	NbparT=5
	TrainGenerator=gd.GenDataSet(NbB=NbparT)
	TestGenerator=gd.GenDataSet(mint=(10.5,50.5))
	
	##Parametres du training
	batch_size = 250
	tabEpochs =[4,8,20,40,100]#[ int(10*(i**1.2)) for i in range(NbSets+1)]
	lr=3e-3
	tabSNR=[36,24,16,12,8] 

	##Trainer
	mytrainer=tr.MyTrainer(tabSNR=tabSNR ,tabEpochs=tabEpochs,lr=lr,batch_size=batch_size)
	
	##Results
	SNRtest=10
	myresults=ur.Results(TestGenerator,SNRtest=SNRtest)
	
	#Training
	mytrainer.train(TrainGenerator,myresults)
	
def testTrainerIO():
	chemintrain=os.path.dirname(__file__)+'/../generators/(10.0, 50.0)-5-(0.75, 0.95)-flat-1.0-2048.0.p'
	chemintest=os.path.dirname(__file__)+'/../generators/(10.5, 50.5)-1-(0.75, 0.95)-flat-1.0-2048.0.p'
	print("Chargement des DataSets chemin: ",chemintrain," et : ",chemintest)
	TrainGenerator=gd.GenDataSet.readGenerator(chemintrain)
	TestGenerator=gd.GenDataSet.readGenerator(chemintest)
	print("Chargement DataSets Ok")
	##Parametres du training
	cheminparams=os.path.dirname(__file__)+'/../params/default_trainer_params.csv'

	##Trainer
	mytrainer=tr.MyTrainer(paramFile=cheminparams)
	
	##Results
	SNRtest=10
	myresults=ur.Results(TestGenerator,SNRtest=SNRtest)
	
	#Training
	mytrainer.train(TrainGenerator,myresults)
	
	##saving results
	cheminresults=os.path.dirname(__file__)+'/../results/'
	myresults.saveResults(cheminresults)

if __name__=="__main__":	
	#testTrainer()
	testTrainerIO()
	
