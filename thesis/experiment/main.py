from msc import config
from msc.experiments import OfflineExperiment

def main():
    print("Beginning experiment")

if __name__ == "__main__":
    
    experiment = OfflineExperiment(config)
    
    results = experiment.run()
    
    
    
    