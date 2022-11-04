from DGWS import useResults as ur
import matplotlib.pyplot as plt
import os

def test():
	cheminload=os.path.dirname(__file__)+'/../results/DecrSca-flat-100-250-0.003-8.0.p'
	cheminload2=os.path.dirname(__file__)+'/../results/DecrSca-flat-100-250-0.03-8.0.p'
	result=ur.Results.readResults(cheminload)
	result2=ur.Results.readResults(cheminload2)
	#définition du Printer
	printer=ur.Printer()
	
	printer.plotDistrib(result,0)
	printer.plotDistrib(result,25)
	printer.plotDistrib(result,50)
	printer.plotDistrib(result,75)
	printer.plotDistrib(result,99)
	printer.plotMapDistrib(result,99)
	printer.plotROC(result)
	printer.plotSensitivity([result,result2])
	
	## Sauvegarde des figures
	cheminsave=os.path.dirname(__file__)+'/../prints/'
	printer.savePrints(cheminsave)
	plt.show()


if __name__ == "__main__":
	test()
