


class OfflineExperiment:
    """ This class orchestrates the offline experiment from start to finish.
    The offline eperiment consists of:
    * collect dataset
    * save & load data to disk               # perhaps hdf5
    * transform dataset z(x) for all x       # GP embedding transformation
    * estimate density p(z)                  # GMM
    * extract novelty score n(x)             # p-value
    * init prior p(S)                        # PyroModule
    * dump state
    
    """
    def __init__(self, config):
      self.config = config
      self.ds = None
      self.results = None
      
    def get_dataset(self):
        # get dataset from iEEG.org
        pass
    
    def analyze_results(self):
        assert self.results is not None, "error: self.results is None"
        results = self.results

    
    def run(self):
        # run experiment
        results = None
        analyze_results(results)
        return results
    




class OnlineExperiment:
    """ This class orchestrates the online experiment from start to finish.
    The online eperiment consists of:
    * load state
    * init times
    * for t in times:
    * init x_t = ds[t]
    * estimate p(S_t)
    * estimate n(x_t)
    * multiply p(S|X) = n(x_t) * p(S_t)
    * save to results.csv 
    """
    def __init__(self) -> None:
        pass

