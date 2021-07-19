import ROOT
import uproot
import glob
import pprint
import matplotlib.pyplot as plt
import sys
from os import path
from pathlib import Path
from tqdm import tqdm

class CutFunction:
    def __init__(self, pfCand_entry: str, pfCand_entryDatatype: str, newEntryName: str, deltaR_threshold: float=0.5):
        self.functionName = "cut_" + pfCand_entry
        self.pfCand_entryDatatype = pfCand_entryDatatype
        self.pfCand_entry = pfCand_entry
        self.deltaR_threshold = deltaR_threshold
        self.code = self.generateCppFunc_CutByPuppiWeightNoLep()
        if self.pfCand_entry == 'pfCand_PuppiWeightNoLep' or self.pfCand_entry == 'pfCand_DeltaR':
            self.call = self.functionName + f"(pfCand_PuppiWeightNoLep, pfCand_DeltaR)"
        else:
            self.call = self.functionName + f"({self.pfCand_entry}, pfCand_PuppiWeightNoLep, pfCand_DeltaR)"
        self.newEntryName = newEntryName

    def generateCppFunc_CutByPuppiWeightNoLep(self) -> str:
        if self.pfCand_entry == 'pfCand_PuppiWeightNoLep' or self.pfCand_entry == 'pfCand_DeltaR':
            return f"""
            std::vector<{self.pfCand_entryDatatype}> {self.functionName} (
                const ROOT::VecOps::RVec<float>& pfCand_PuppiWeightNoLep, 
                const ROOT::VecOps::RVec<float>& pfCand_DeltaR){{

                std::vector<{self.pfCand_entryDatatype}> v;
                for(std::size_t i=0; i < {self.pfCand_entry}.size(); i++){{
                    if(pfCand_PuppiWeightNoLep[i] > 0.0 && pfCand_DeltaR[i] < {self.deltaR_threshold}){{
                        v.push_back({self.pfCand_entry}[i]);
                    }}
                }}
                return v;
            }};
            """
        else:
            return f"""
            std::vector<{self.pfCand_entryDatatype}> {self.functionName} (
                const ROOT::VecOps::RVec<{self.pfCand_entryDatatype}>& {self.pfCand_entry}, 
                const ROOT::VecOps::RVec<float>& pfCand_PuppiWeightNoLep, 
                const ROOT::VecOps::RVec<float>& pfCand_DeltaR){{

                std::vector<{self.pfCand_entryDatatype}> v;
                for(std::size_t i=0; i < {self.pfCand_entry}.size(); i++){{
                    if(pfCand_PuppiWeightNoLep[i] > 0.0 && pfCand_DeltaR[i] < {self.deltaR_threshold}){{
                        v.push_back({self.pfCand_entry}[i]);
                    }}
                }}
                return v;
            }};
            """

cutFunctions = { 
    CutFunction("pfCand_Pt", "float", "pfCand_pt"),
    CutFunction("pfCand_Eta", "float", "pfCand_eta"),
    CutFunction("pfCand_Phi", "float", "pfCand_phi"),
    CutFunction("pfCand_Mass", "float", "pfCand_mass"),
    CutFunction("pfCand_Charge", "int32_t", "pfCand_charge"),
    CutFunction("pfCand_ParticleType", "int32_t", "pfCand_particleType"),
    CutFunction("pfCand_JetDaughter", "int32_t", "pfCand_jetDaughter"),
    CutFunction("pfCand_PuppiWeightNoLep", "float", "pfCand_puppiWeightNoLep"),
    CutFunction("pfCand_DeltaPhi", "float", "pfCand_deltaPhi"),
    CutFunction("pfCand_DeltaEta", "float", "pfCand_deltaEta"),
    CutFunction("pfCand_DeltaR", "float", "pfCand_deltaR"),
}

baseInputFolder = '/ceph/aissac/ntuple_for_graphs/prod_2018_v2_processed_v5'
files = glob.glob(baseInputFolder + "/trimmed_500000_and_added_deltaPhiEtaR" + "/**/*.root", recursive=True)
treename = "taus"
n = 200000

outputFolder = path.join(baseInputFolder, f'trimmed_{n}_and_cut_puppiWeightNoLep_greater_0_and_deltaR_smaller_0point5')
Path(outputFolder).mkdir(parents=True, exist_ok=True)

branchList = ROOT.vector('string')()
# only declare c++ code once, not for every file
for f in cutFunctions:
    ROOT.gInterpreter.Declare(f.code)
    branchList.push_back(f.newEntryName)

for file in tqdm(files):
    outputFileName = path.join(outputFolder, file.split('/')[-1])
    df = ROOT.RDataFrame(treename, file)

    for f in cutFunctions:
        df = df.Define(f.newEntryName, f.call)

    df = df.Range(n)
    df.Snapshot(treename, outputFileName, branchList)

