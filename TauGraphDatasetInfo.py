from PPrintable import PPrintable
from JsonSerializable import JsonSerializable


class TauGraphDatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, graphClassMap: dict, rootfiles: dict,
    nodeFeatMap: dict, edgeFeatMap: dict, graphFeatMap: dict, 
    splitPercentages: dict, graphsPerClass: int):
        self.name = name
        self.rootfiles = rootfiles
        self.rootfilesCount = 0
        for files in self.rootfiles.values():
            self.rootfilesCount += len(files)

        self.graphClassMap = graphClassMap
        self.graphsPerClass = graphsPerClass
        self.numberOfGraphClasses = len(rootfiles.keys())
        self.nodeFeatMap = nodeFeatMap
        self.edgeFeatMap = edgeFeatMap
        self.graphFeatMap = graphFeatMap
        self.splitPercentages = splitPercentages
