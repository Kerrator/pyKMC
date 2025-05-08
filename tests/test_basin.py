from pykmc import System, Config
import os

class TestBassin : 


    def test_detection(self, config: Config)  : 


        DIR = os.path.dirname(__file__)
        config_path = os.path.join(DIR, 'data', 'CuSia+Vac.xyz')
        print(config_path)
        #Initialize System avec fichier xyz 
        #Initialize Catalog avec pickle qui contient que l'évènement flicker 
        #Calculer atomic enviromnet et ajouter à visited_environement -> la KMC ne pourra faire que des flickers 
        #Lancer autant de pas que de threshold évènements passé identiques 
        #assert si bassin détécté. 

