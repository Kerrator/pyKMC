# pyKMC

Pour le moment j'ai une classe par fonctionnalité mais je suis pas certain que ça soit une bonne idée.
Il faudrait peut être faire en sorte d'etendre l'objet Atoms de ASE avec les paramètres nécessaires.
Ensuite : 
```
Atoms.minimize('lammps')
Atoms.atomiv_env('cna')
Atoms.search_event(catalog)
```
un truc du genre

in ./utils :

- generate_initial_configs : 
scripts to generate example initial configurations 

- visualization : 
scripts to visualize similar atomic environment search
