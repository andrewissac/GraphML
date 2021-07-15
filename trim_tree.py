import ROOT
import uproot
import glob
import pprint
import matplotlib.pyplot as plt
import sys
from os import path


files = glob.glob("/ceph/akhmet/forAndrewIsaac/prod_2018_v2_processed_v4" + "/**/*.root", recursive=True)
treename = "taus"

testfile = '/ceph/akhmet/forAndrewIsaac/prod_2018_v2_processed_v4/DYJetsToLL_M-50_jets.root'

df = ROOT.RDataFrame(treename, testfile)

class FilterFunction:
    def __init__(self, pfCand_entry: str, pfCand_entryDatatype: str, code: str=""):
        self.functionName = "filter_" + pfCand_entry
        self.pfCand_entryDatatype = pfCand_entryDatatype
        self.pfCand_entry = pfCand_entry
        if(code == ""):
            self.code = self.generateCppFunc_FilterByPuppiWeightNoLep()
            self.call = self.functionName + f"({self.pfCand_entry},pfCand_puppiWeightNoLep)"
        else:
            self.code = code
            self.call = self.functionName + f"({self.pfCand_entry})"
        self.newEntryName = pfCand_entry + "_cut"

    def generateCppFunc_FilterByPuppiWeightNoLep(self) -> str:
        return f"""
        std::vector<{self.pfCand_entryDatatype}> {self.functionName} (const ROOT::VecOps::RVec<{self.pfCand_entryDatatype}>& {self.pfCand_entry}, const ROOT::VecOps::RVec<int32_t>& pfCand_puppiWeightNoLep){{
            std::vector<{self.pfCand_entryDatatype}> v;
            for(std::size_t i=0; i < {self.pfCand_entry}.size(); i++){{
                if(pfCand_puppiWeightNoLep[i] > 0){{
                    v.push_back({self.pfCand_entry}[i]);
                }}
            }}
            return v;
        }};
        """

class CalculateFunction:
    def __init__(self, newEntryName: str, call: str, code: str):
        self.newEntryName = newEntryName
        self.call = call
        self.code = code

CalcFunctions = [
    CalculateFunction(
        "pfCand_deltaPhi", 
        "GetDeltaPhi(pfCand_phi, tau_phi)",
        f"""
        float const kPI = TMath::Pi();
        float const kTWOPI = 2.*kPI;

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
        "pfCand_deltaEta", 
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
        "pfCand_deltaR", 
        "GetDeltaR(pfCand_phi, pfCand_eta, tau_phi, tau_eta)",
        f"""
        std::vector<float> GetDeltaR (const ROOT::VecOps::RVec<float>& pfCand_phi, const ROOT::VecOps::RVec<float>& pfCand_eta, float tau_phi, float tau_eta){{
            std::vector<float> v;
            for(std::size_t i=0; i < pfCand_eta.size(); i++){{
                float deltaPhi = TVector2::Phi_mpi_pi(tau_phi - pfCand_phi[i]); 
                float deltaEta = tau_eta - pfCand_eta[i]; 
                float deltaR = TMath::Sqrt(deltaPhi * deltaPhi + deltaEta * deltaEta);
                v.push_back(deltaR);
            }}
            return v;
        }};
        """)
]

ff = { 
    'pfCand_pt': FilterFunction("pfCand_pt", "float"),
    'pfCand_eta': FilterFunction("pfCand_eta", "float"),
    'pfCand_phi': FilterFunction("pfCand_phi", "float"),
    'pfCand_mass': FilterFunction("pfCand_mass", "float"),
    'pfCand_charge': FilterFunction("pfCand_charge", "int32_t"),
    'pfCand_particleType': FilterFunction("pfCand_particleType", "int32_t"),
    'pfCand_puppiWeightNoLep': FilterFunction(
        "pfCand_puppiWeightNoLep",
        "int32_t",
        f"""
        std::vector<int32_t> filter_pfCand_puppiWeightNoLep (const ROOT::VecOps::RVec<int32_t>& pfCand_puppiWeightNoLep){{
            std::vector<int32_t> v;
            for(std::size_t i=0; i < pfCand_puppiWeightNoLep.size(); i++){{
                if(pfCand_puppiWeightNoLep[i] > 0){{
                    v.push_back(pfCand_puppiWeightNoLep[i]);
                }}
            }}
            return v;
        }};
        """
        )
}

branchList = ROOT.vector('string')()
for key, val in ff.items():
    #print(key + ": " + val.code + "\n" + val.call + "\n" + "#####################")
    ROOT.gInterpreter.Declare(val.code)
    branchList.push_back(val.newEntryName)
    df = df.Define(val.newEntryName, val.call)

for f in CalcFunctions:
    ROOT.gInterpreter.Declare(f.code)
    branchList.push_back(f.newEntryName)
    df = df.Define(f.newEntryName, f.call)

n = 250000
outputFolder = '/ceph/aissac/ntuple_for_graphs/prod_2018_v2_processed_v4_puppiWeightNoLep_greater_0_cut'
outputFileName = path.join(outputFolder, testfile.split('/')[-1].replace('.root', f'_trimmed_n_{n}.root'))

df_cut = df.Range(n)
df_cut.Snapshot(treename, outputFileName, branchList)

