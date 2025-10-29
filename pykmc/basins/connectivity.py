import pandas as pd

class StatesConnectivity() : 

    def __init__(self) : 
        """Connectivity between system states"""

        self.df = pd.DataFrame(columns=['state', 
                                        'state_connexion', 
                                        'event_connexion',
                                        'central_atom', 
                                        'sym',
                                        'transient',
                                        'dE', 
                                        'k']) 
        self.graph = None # Placeholder for later if needed to generate a connectivity graph from the dataframe, for path finding and vizualization

    def add_connectivity(self, state, state_connexion, event_connexion, central_atom, sym, transient, dE, k )  : 
        new_row = pd.DataFrame([{'state': state, 
                                 'state_connexion': state_connexion, 
                                 'event_connexion': event_connexion, 
                                 'central_atom': central_atom, 
                                 'sym': sym, 
                                 'transient': transient, 
                                 'dE': dE,
                                 'k': k}])
        self.df = pd.concat([self.df, new_row], ignore_index=True)


    def get_transition_to_state(self, target_state:int, as_tuples: bool = True, return_all:bool = False) : 

        """Return the transition(s) leading to the specified target state.

        This method filters the internal connectivity dataframe to find all transitions
        that reach `target_state`. If `return_all` is False, only the transition
        starting from the smallest `state` is returned.

        Parameters
        ----------
        target_state : int
            Identifier of the target state.
        as_tuples : bool, optional (default=True)
            If True, return the transitions as a list of tuples 
            (state, event_connexion, central_atom, sym). 
            If False, return a sub pandas DataFrame of the connectivity DataFrame
        return_all : bool, optional (default=False)
            If True, return all possible transitions to `target_state`.
            If False, return only the first transition starting from the smallest `state`.

        Returns
        -------
        tuple or list[tuple] or pd.DataFrame
            - If `as_tuples` is True:
                - If `return_all` is False: returns a list containing a single tuple 
                  (state, event_connexion, central_atom, sym) corresponding to the
                  transition from the smallest state.
                - If `return_all` is True: returns a list of tuples for all matching transitions.
            - If `as_tuples` is False:
                - Returns a pandas DataFrame containing the matching row(s).
        """

        #Find sub dataframe with state_connexion == target_state 
        sub_df = self.df[self.df['state_connexion'] == target_state]
        
        if not return_all:
            #Return the first transition with lower from state
            min_state = sub_df['state'].min()
            sub_df = sub_df[sub_df['state'] == min_state].iloc[[0]]  # only first row 
            return self.to_tuples(sub_df)[0] if as_tuples else sub_df
        else : 
            return self.to_tuples(sub_df) if as_tuples else sub_df


    

    def get_table(self) : 
        return self.df 
    

    def to_tuples(self, df:pd.DataFrame):
        """Convert a DataFrame to a list of tuples.

        Parameters
        ----------
        df : pd.DataFrame, optional
            The DataFrame to convert. If None, uses the full internal table.

        Returns
        -------
        list[tuple]
            Each tuple contains (state, event_connexion, central_atom, sym).
        """
        return list(df[['state', 'event_connexion', 'central_atom', 'sym', 'transient']].itertuples(index=False, name=None))
    

    def reorder_states_index(self) : 
        """If the connectivity table has non continuous state index, reorder them"""

        unique_states = sorted(set(self.df["state"]) | set(self.df["state_connexion"]))

        mapping = {old: new for new, old in enumerate(unique_states)}

        # Étape 3 : appliquer au DataFrame
        df_new = self.df.copy()
        df_new["state"] = df_new["state"].map(mapping)
        df_new["state_connexion"] = df_new["state_connexion"].map(mapping)

        self.df = df_new 

        return  mapping
    
    def save(self, outfile: str = "basin_connectivity.pickle") -> None:
        """Save the connectivity DataFrame to a pickle file.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'basin_connectivity.pickle'.

        """
        self.df.to_pickle(outfile)

    def clear(self) : 
        """Clear the connectivity DataFrame"""
        self.df = pd.DataFrame(columns=['state', 
                                        'state_connexion', 
                                        'event_connexion',
                                        'central_atom', 
                                        'sym',
                                        'transient', 
                                        'dE', 
                                        'k'])
        

class BasinStatesConnectivity(StatesConnectivity)  : 
    """Connectivity table with basin exploration specific operations"""
    
    def change_state_index(self, current_index:int, new_index:int) : 
        """Update all occurrences of a state index in the connectivity DataFrame.

        This method is used during basin exploration when a state that needs to be 
        explored is found to be identical to a previously explored state. All rows 
        in the connectivity DataFrame where `state` or `state_connexion` equals `current_index` are updated 
        to `new_index`.

        Parameters
        ----------
        current_index : int
            The state index to be replaced.
        new_index : int
            The new state index to assign.
        """
        self.df.loc[self.df['state'] == current_index, 'state'] = new_index
        self.df.loc[self.df['state_connexion'] == current_index, 'state_connexion'] = new_index

    def change_state_to_absorbing(self, state_connexion) : 
        """ """
        self.df.loc[self.df['state_connexion']== state_connexion, 'transient'] = False

    def merge(self, states_connectivity: "StatesConnectivity") : 
        """Merge another StatesConnectivity's DataFrame into this connectivity table.

        This method appends the DataFrame from `states_connectivity` to the internal
        connectivity table (`self.df`) of the current object. It is typically used
        during basin exploration when a new state has been explored.

        Parameters
        ----------
        states_connectivity : StatesConnectivity
            Another StatesConnectivity instance whose connectivity DataFrame (`df`)
            will be appended to this object's DataFrame.

        Notes
        -----
        - The index of the resulting DataFrame is reset.
        - This operation does not remove duplicates; ensure uniqueness if required.
        """

        self.df = pd.concat([self.df, states_connectivity.df], ignore_index=True)