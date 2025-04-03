import pypARTn2
import numpy as np


def pARTn_search(lmp, config_event_search, central_atom_idx) : 
    #Parameters : 
    rcutenv = config_event_search['rcutenv']
    #INITILIZE ARTN
    artn = pypARTn2.artn(engine='lmp')

    #LAMMPS COMMANDS
    lmp.command("plugin load {}".format(config_event_search['path_artnso']))
    lmp.command("fix 10 all artn dmax {}".format(config_event_search['partn_dmax']))
    lmp.command("min_style fire")

    #SETUP ARTN
    artn.set('engine_units', 'lammps/metal')
    artn.set('verbose',config_event_search['partn_verbose'])
    artn.set('struc_format_out', 'none')
    artn.set("lpush_final", True)
    artn.set("lmove_nextmin", False) #if true fortran runtime error when event not found
    artn.set("ninit", config_event_search['partn_ninit'])
    artn.set("forc_thr", config_event_search['partn_forc_thr'])
    artn.set('push_mode', config_event_search['partn_push_mode'])
    if config_event_search['partn_push_mode'] == 'rad' :
        artn.set('push_dist_thr', config_event_search['partn_push_dist_thr'])
    artn.set("push_step_size",  config_event_search['partn_push_step_size'])
    artn.set("push_ids", [central_atom_idx])
    artn.set('eigen_step_size', config_event_search['partn_eigen_step_size'])
    artn.set('lanczos_disp', config_event_search['partn_lanczos_disp'])
    artn.set('nsmooth',  config_event_search['partn_nsmooth'])
    artn.set('nperp', config_event_search['partn_nperp'])

    #RUN
    lmp.command("minimize 1e-6 1e-8 1000 1000")

    #EXTRACT DATA
    err = artn.get_runparam("error_message")
    if not err :
        #Results
        delr1 = artn.extract('delr_min1')
        delr2 = artn.extract('delr_min2')
        #Checks if one minimum is close to the original configuration
        if delr1 < config_event_search['partn_delr_threshold'] or delr2 < config_event_search['partn_delr_threshold'] :
            E_sad = artn.extract("etot_sad")
            E_min1 = artn.extract("etot_min1")
            E_min2 = artn.extract("etot_min2")

            dE_forward = E_sad - E_min1
            dE_backward = E_sad - E_min2

            min1positions = artn.extract("tau_min1")
            min2positions = artn.extract("tau_min2")
            saddlepositions = artn.extract("tau_sad")

            #find atom that moves the most 
            dist = (min1positions-saddlepositions)**2
            dist = dist.sum(axis=-1)
            dist = np.sqrt(dist)
            dist[dist > rcutenv] = 0 #if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
            index_move = np.argmax(dist)
            return min1positions, saddlepositions, min2positions, index_move, dE_forward, dE_backward
        else :
            return None
    else :
        return None
