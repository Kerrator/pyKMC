Priority : 
- [ ] Make check distance dh in update position a parameter. 
- [ ] Check what happen when dh distance is high 
- [ ] Topology graph with different chemical elements (add vertex coloring)
- [ ] Deal with symmetries 
- [ ] Debug log and put pARTn fails in file


##### General
- Pour le moment, psr utilise que IRA et retourne rmat, tr, perm et dh. Si un jour on veut utiliser autre chose, il faut faire des modifications pour que �a soit plus g�n�ral (surtout sur le Return de la fonction system.psr())
- Voir comment g�rer les potentiels Lammps si on a autre chose que des pair_style (eg pair_bonds) -> peut être créer une fonction dans utilities
- Idem, pour le moment on utilise que lammps, si un jour on veut faire des calculs de forces avec un autre code, il faudra modifier comment est g�r� la partie potentiel
- Quand on utilise Lammps, j'utilise des commandes hardcodé(boundary, units metal, atom_style atomic ...) il se peut que dans le suite il faille changer aussi �a pour que �a soit plus g�n�ral.
- Utiliser des verlets/cell lists
- make an executable (so we can do : pykmc -in input.in)
    - check nuitka python module
- add verbosity and debug log level
- maybe using generator instead of list can improve some perfomance
- En fait je crois que c'est stupide de check la topologie a la reconstruction puisque comme je prends les positions final de l'evenement, applique transformation matrices, et remplace ces positions dans le system, si apres je recalcul la topologie baaaah c'est deux fois le meme calcul. Avec IRA, la reconstruction est pas fait de la meme manière qu'avec kart. A la limite calculer dE ok, si on a lors de la reconstructoin les positions dans rcutevent sont legerement deplacé mais meme ça, si dh de ira est vriament faible, ça devrait pas être nécessaire, puisque, encore une fois, on va calculer la meme chose. 
- Comme on utilise les positions finales données par partn qui sont relaxées, est ce qu'on a besoin de relaxé ? j'imagine que ça dépend de dh 
- Add restart option --> if yes not use configuration file first step
- Could think of a when to reuse a catalog with different temperature, meaning need to recompute k
- Topo reconstruction pourrait devenir un chekc warning pour dire qu'il faut augmenter rcut
- Quand SIA va dans VAC, tous les ID deviennent "crystal" et donc le code crash parce y'a plus d'environement ID dans le catalog (crash a la fonction selected event rejection free, cad que len(k) == 0 ) -> il faut faire une sortie de la boucle ici avec un message dans le log


##### config
- Dans config, on devrait checker si les param�tres de inputs sont coh�rents
##### Atomic environment 
- dans cna_graph, il faudrait adapter pour utiliser cna puis graph, et pas tous reecrire.
- better gather results 
- See if we can improve the way I connect in make_graph the graph atom index and system atom index (I think that some part are not necessary)
##### Event search 
- could add dimer and neb (with ASE)
- for pARTn parameter, should search every items in config that start with 'ira' and set them, else default partn parameter
- for reconstruction == False, use add_event_without_reconstruction to check if event already in the catalog. To check, use np.allclose() method. It checks, for a same atom_index, if the final positions are close. np.allclose() use tolerance parameters, for the moment this is hardcorded, will need to see what are the best parameters, and if we use default hardcoded values or put it in the input file
- What is the value that we should use for the condition delr1 < 0.2 or delr2 < 0.2
- Need to clean the selection of the atom that move the most and the finding of the neighbor list but will be better to do it after implementing verlet/cell list

##### PSR : 
- Il faudrait voir comment paralleliser IRA, je suis pas certain que ça améliore les performances (demander à Miha).
- Je considère que si on a les meme topology alors la list des types sont identiques (voir typ1=typ2). Est ce que c'est vrai ? to check with alloy

###### KMC : 
- if restart dont save first configuration at step = 0
- voir check_topo (pour le moment changer en return True)
- reconstruction avec pARTn