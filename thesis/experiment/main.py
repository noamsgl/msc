from msc import config

class OfflineExperiment:
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
        return results
    

def main():
    print("Hello world")

if __name__ == "__main__":
    experiment = OfflineExperiment(config)
    results = experiment.run()
    analyze_results(results)