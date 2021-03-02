import datajoint as dj

schema = dj.schema('tutorial', locals())       # this might differ depending on how you setup

@schema
class Mouse(dj.Manual):
      definition = """
      # mouse
      mouse_id: int                  # unique mouse id
      ---
      dob: date                      # mouse date of birth
      sex: enum('M', 'F', 'U')    # sex of mouse - Male, Female, or Unknown/Unclassified
      """
        
        
@schema 
class Session(dj.Manual): 
    definition = """
    # experiment session
    -> Mouse                     # depends on Mouse table
    session_date: date           # session date
    ---
    experiment_setup: int        # experiment setup ID
    experimenter: varchar(128)   # name of the experimenter
    """
    
        
@schema 
class Neuron(dj.Imported): 
    definition = """
    # measure from sessions
    -> Session
    ---
    activity: longblob    # electric activity of the neuron
    """
    
    def _make_tuples(self, key):    # _make_tuples takes a single argument `key` (a dictionary)
        # use key dictionary to determine the data file path
       data_file = path_to_data + "data_{mouse_id}_{session_date}.npy".format(**key)

       # load the data
       data = np.load(data_file)

       # add the loaded data as the "activity" column
       key['activity'] = data

       # insert the key into self
       self.insert1(key)

       print('Populated a neuron for {mouse_id} on {session_date}'.format(**key))