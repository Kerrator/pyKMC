# KMC Algorithm Overview 

To implement
- Symmetry : Il faudrait quand on ajoute l'évènement générique au catalog, regarder toutes les symmétries avec SOFI. Et on ajoute une colonne au catalog avec les symmetries. 
- Specific event 
- Basin 



1. Initialization 
    - Log 
    - System
    - Engine 
    - Minimization with engine and update positions System 
    - NeighborsList 
    - AtomicEnvironment 
    - Catalog 

    - visited environment (set('crystal')) 

    - time = 0 
    - append snapshot to trajectory

Enter KMC LOOP 

2. Find new non visited environments 
    - difference between visited environment and atomic environment hash list
3. For each new environements, launch `nsearch` event searches 
    - find nsearch atoms having the correspond ID 
    - launch event search with engine
    - if reconstruction == True, recenter event (to deal with pbc)
    - add to catalog    
        - **For symmetries** (see sofi.py in tests/SOFI) : 
            - use SOFI to find all symmetriea of initial positions 
            - compute displacement matrix between initial and finale event positions 
            - apply symmetries to displacement vector -> conserve only symmetries that correspond to a unique displacement 
            - save symmetries in catalog (need to add column, save matrix and perm)
**CHANGE** 
4. Select event : 
    - create new specific catalog 
    - for all generic event : 
        - if it contribute to 99.9% ktot : 
            - for all atoms having the same graph id : 
                - raffine event and its symmetric  
                - add to specific catalog
        - else add generic event and symmetric for all atom having the same graph id
    - rejection free algo to select event

**CHANGE** 
5. If it enter bassin : 
    guess it depends on the method
    algo basin to modify spécific 


**CHANGE** 
6. Apply event : 
    - update position based on final position selected event

6. Update neighbors list and atomic environment 


