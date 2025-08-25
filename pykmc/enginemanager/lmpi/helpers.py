def initialize_parameters(session) : 
    session.command("units metal")
    session.command("atom_style atomic")
    session.command("dimension 3")
    session.command("boundary p p p")
    session.command("atom_modify sort 0 0.0")

#def initialize_system(session, system) : 

