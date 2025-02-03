##### General
- Pour le moment, psr utilise que IRA et retourne rmat, tr, perm et dh. Si un jour on veut utiliser autre chose, il faut faire des modifications pour que √a soit plus g√n√ral (surtout sur le Return de la fonction system.psr())
- Check lammps create_atoms commande au lieu de write des fichiers d'inputs
- Voir comment g√rer les potentiels Lammps si on a autre chose que des pair_style (eg pair_bonds)
- Idem, pour le moment on utilise que lammps, si un jour on veut faire des calculs de forces avec un autre code, il faudra modifier comment est g√r√ la partie potentiel
- Quand on utilise Lammps, j'utilise des commandes hardcod√©(boundary, units metal, atom_style atomic ...) il se peut que dans le suite il faille changer aussi √a pour que √a soit plus g√n√ral.
- Utiliser des verlets/cell lists

##### config
- Dans config, on devrait checker si les param√tres de inputs sont coh√rents
##### Atomic environment 
- dans cna_graph, il faudrait adapter pour utiliser cna puis graph, et pas tous reecrire.
- better gather results 
- See if we can improve the way I connect in make_graph the graph atom index and system atom index (I think that some part are not necessary)
