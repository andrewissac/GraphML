import ROOT
import uproot
import glob
import pprint
import matplotlib.pyplot as plt
import sys
from os import path
from pathlib import Path
from tqdm import tqdm

"""
THIS IS TO REDUCE THE NUMBER OF GRAPHS TO N = 500000
ALSO ADDS DELTA PHI/ETA/R
"""

class RVecToVectorFunction:
    def __init__(self, pfCand_entry: str, pfCand_entryDatatype: str, newEntryName: str):
        self.functionName = "convert_" + pfCand_entry
        self.pfCand_entryDatatype = pfCand_entryDatatype
        self.pfCand_entry = pfCand_entry
        self.code = self.generateCppFunc()
        self.call = self.functionName + f"({self.pfCand_entry})"
        self.newEntryName = newEntryName

    def generateCppFunc(self) -> str:
        return f"""
        std::vector<{self.pfCand_entryDatatype}> {self.functionName} (const ROOT::VecOps::RVec<{self.pfCand_entryDatatype}>& {self.pfCand_entry}){{
            std::vector<{self.pfCand_entryDatatype}> v;
            for(std::size_t i=0; i < {self.pfCand_entry}.size(); i++){{
                v.push_back({self.pfCand_entry}[i]);
            }}
            return v;
        }};
        """

class CalculateFunction:
    def __init__(self, newEntryName: str, call: str, code: str):
        self.newEntryName = newEntryName
        self.call = call
        self.code = code

calcFunctions = [
    CalculateFunction(
        "pfCand_DeltaPhi", 
        "GetDeltaPhi(pfCand_phi, tau_phi)",
        f"""
        std::vector<float> GetDeltaPhi (const ROOT::VecOps::RVec<float>& pfCand_phi, float tau_phi){{
            std::vector<float> v;
            for(std::size_t i=0; i < pfCand_phi.size(); i++){{
                float deltaPhi = TVector2::Phi_mpi_pi(tau_phi - pfCand_phi[i]); 
                v.push_back(deltaPhi);
            }}
            return v;
        }};
        """
        ),
    CalculateFunction(
        "pfCand_DeltaEta", 
        "GetDeltaEta(pfCand_eta, tau_eta)",
        f"""
        std::vector<float> GetDeltaEta (const ROOT::VecOps::RVec<float>& pfCand_eta, float tau_eta){{
            std::vector<float> v;
            for(std::size_t i=0; i < pfCand_eta.size(); i++){{
                float deltaEta = tau_eta - pfCand_eta[i]; 
                v.push_back(deltaEta);
            }}
            return v;
        }};
        """),
    CalculateFunction(
        "pfCand_DeltaR", 
        "GetDeltaR(pfCand_DeltaPhi, pfCand_DeltaEta)",
        f"""
        std::vector<float> GetDeltaR (
            const ROOT::VecOps::RVec<float>& pfCand_DeltaPhi, 
            const ROOT::VecOps::RVec<float>& pfCand_DeltaEta){{

            std::vector<float> v;
            for(std::size_t i=0; i < pfCand_DeltaPhi.size(); i++){{
                float deltaR = TMath::Sqrt(pfCand_DeltaPhi[i] * pfCand_DeltaPhi[i] + pfCand_DeltaEta[i] * pfCand_DeltaEta[i]);
                v.push_back(deltaR);
            }}
            return v;
        }};
        """)
]

convertFunctions = [ 
    RVecToVectorFunction("pfCand_pt", "float", "pfCand_Pt"),
    RVecToVectorFunction("pfCand_eta", "float", "pfCand_Eta"),
    RVecToVectorFunction("pfCand_phi", "float", "pfCand_Phi"),
    RVecToVectorFunction("pfCand_mass", "float", "pfCand_Mass"),
    RVecToVectorFunction("pfCand_charge", "int32_t", "pfCand_Charge"),
    RVecToVectorFunction("pfCand_particleType", "int32_t", "pfCand_ParticleType"),
    RVecToVectorFunction("pfCand_jetDaughter", "int32_t", "pfCand_JetDaughter"),
    RVecToVectorFunction("pfCand_puppiWeightNoLep", "float", "pfCand_PuppiWeightNoLep")
]

files = glob.glob("/ceph/akhmet/forAndrewIsaac/prod_2018_v2_processed_v5" + "/**/*.root", recursive=True)
treename = "taus"
n = 500000

outputFolder = path.join('/ceph/aissac/ntuple_for_graphs/prod_2018_v2_processed_v5', f'trimmed_{n}_and_added_deltaPhiEtaR')
Path(outputFolder).mkdir(parents=True, exist_ok=True)

branchList = ROOT.vector('string')()
# only declare c++ code once, not for every file
for f in calcFunctions:
    ROOT.gInterpreter.Declare(f.code)
    branchList.push_back(f.newEntryName)

for f in convertFunctions:
    ROOT.gInterpreter.Declare(f.code)
    branchList.push_back(f.newEntryName)

for file in tqdm(files):
    outputFileName = path.join(outputFolder, file.split('/')[-1])
    df = ROOT.RDataFrame(treename, file)

    for f in calcFunctions:
        df = df.Define(f.newEntryName, f.call)

    for f in convertFunctions:
        df = df.Define(f.newEntryName, f.call)

    df = df.Range(n)
    df.Snapshot(treename, outputFileName, branchList)

