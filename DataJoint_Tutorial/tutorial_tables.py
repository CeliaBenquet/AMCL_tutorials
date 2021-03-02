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
        
@schema
class ActivityStatistics(dj.Computed):
    definition = """
    -> Neuron
    ---
    mean: float    # mean activity
    stdev: float   # standard deviation of activity
    max: float     # maximum activity
    """

    def _make_tuples(self, key):
        activity = (Neuron() & key).fetch1('activity')    # fetch activity as NumPy array

        # compute various statistics on activity
        key['mean'] = activity.mean()   # compute mean
        key['stdev'] = activity.std()   # compute standard deviation
        key['max'] = activity.max()     # compute max
        self.insert1(key)
        print('Computed statistics for mouse_id {mouse_id} session_date {session_date}'.format(**key))
        
@schema
class SpikeDetectionParam(dj.Lookup): # values for a computation rather than raw data
    definition = """
    sdp_id: int       # unique id for spike detection parameter set
    ---
    threshold: float  # threshold for spike detection
    """        
        
@schema
class Spikes(dj.Computed):
    definition = """
    # spikes for a neuron at a certain threshold
    -> Neuron
    -> SpikeDetectionParam
    ---
    spikes: longblob     # detected spikes
    count: int           # total number of detected spikes
    """

    def _make_tuples(self, key):
        print('Populating for: ', key)

        activity = (Neuron() & key).fetch1('activity')
        threshold = (SpikeDetectionParam() & key).fetch1('threshold')

        above_thrs = (activity > threshold).astype(int)   # find activity above threshold
        rising = (np.diff(above_thrs) > 0).astype(int)   # find rising edge of crossing threshold
        spikes = np.hstack((0, rising))    # prepend 0 to account for shortening due to np.diff
        count = spikes.sum()   # compute total spike counts
        print('Detected {} spikes!\n'.format(count))

        # save results and insert
        key['spikes'] = spikes
        key['count'] = count
        self.insert1(key)