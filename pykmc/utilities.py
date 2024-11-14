def modify_lammps_data_2D(lammps_data_file) : 
    """
    Change the line zlo, zhi in a lammps_data_file so zlo/zhi straddle 0.0, ie zlo = -0.5 and zhi = 0.5 
    Usefull for 2D simulation since with ASE we need to add vacuum and so we can't really control the cell parameter
    """
    #TODO should raise error if zlo and zhi not find in file
    zlo = -0.5
    zhi = 0.5

    #Open file
    with open(lammps_data_file, 'r') as file : 
        lines = file.readlines() 

   # Modify the zlo/zhi line : 
    for i, line in enumerate(lines):
        if "zlo" in line and "zhi" in line:
            lines[i] = f"{zlo:.4f} {zhi:.4f} zlo zhi\n"

    #Write new lammps file 
    with open(lammps_data_file, 'w') as file:
        file.writelines(lines) 