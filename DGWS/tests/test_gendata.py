from DGWS import gendata as gd
import matplotlib.pyplot as plt
import os

def testGenNoise():
	##Test class GenNoise
	NGenerator=gd.GenNoise(fe=2048,Ttot=1,kindPSD='analytic')
	
	plt.figure()
	NGenerator.plotPSD(fmin=8)
	NGenerator.getNewSample()
	NGenerator.plotTF(fmin=8)
	
	plt.figure()
	NGenerator.plot()
	NGenerator.getNewSample()
	NGenerator.plot()
	
	plt.legend()
	plt.show()
	
def testGenTemplate():
	##Test class GenTemplate
	TGenerator=gd.GenTemplate(kindTemplate='EOB')
	TGenerator.majParams(50,50)
	TGenerator.getNewSample(kindPSD='analytic')
	
	TGenerator2=gd.GenTemplate(kindTemplate='EM')
	TGenerator2.majParams(50,50)
	TGenerator2.getNewSample(kindPSD='analytic')
	
	plt.figure()
	TGenerator.plotTF()
	TGenerator2.plotTF()
	
	plt.figure()
	TGenerator.plot()
	TGenerator2.plot()
	
	plt.legend()
	plt.show()
	
def testGenDataSet():	
	##Test class GenDataSet
	TrainGenerator=gd.GenDataSet(kindPSD='analytic',NbB=5)
	
	TrainGenerator.plot(-1) #affichage de bruits
	TrainGenerator.plot(-51)
	TrainGenerator.plot(-101)
	plt.show()
	
	
	TrainGenerator.plot(0,100) #affichage de templates bruités à SNRopt fixe
	TrainGenerator.plot(50,100)
	TrainGenerator.plot(100,100)
	plt.show()
	
	
	chemin = os.path.dirname(__file__)+'/../generators/'
	TrainGenerator.saveGenerator(chemin)
	TrainGenerator2=gd.GenDataSet.readGenerator(chemin+'(10, 50)-5-(0.75, 0.95)-analytic-1-2048.p')
	
	TrainGenerator2.plot(0,100) #affichage d'un template à différents SNRopt
	TrainGenerator2.plot(0,1000)
	TrainGenerator2.plot(0,2000)
	plt.show()
	
if __name__=="__main__":
	#testGenNoise()
	testGenTemplate()
	#testGenDataSet()
	
