import uproot
import glob
import pprint

pp = pprint.PrettyPrinter()
files = glob.glob("/ceph/akhmet/forAndrewIsaac/prod_2018_v2_processed" + "/**/*.root", recursive=True)
files = [file + ':taus' for file in files]
pp.pprint(files)

testfile = '/ceph/akhmet/forAndrewIsaac/prod_2018_v2_processed/DYJetsToLL_M-50_jets.root:taus'
ttree = uproot.open(testfile)

test = ttree.keys(filter_name="/pfCand_(pt|eta|phi|mass|charge|particleType)/")
print(test)

print(ttree.show())
pfCand_pt = ttree['pfCand_pt'].array()
# run = ttree['run'].array()
lumi = ttree['lumi'].array()
# evt = ttree['evt'].array()
# pfCand_eta = ttree['pfCand_eta'].array()
# pfCand_phi = ttree['pfCand_phi'].array()
a = 0
b = 5
# print(f'run: {run[a:b]}')
# print(f'evt: {evt[a:b]}')
print(f'lumi: {lumi[a:b]}')
print(f'pfCand_pt: {pfCand_pt[a:b]}')
print([len(x) for x in pfCand_pt[0:200]])
nPfCand = [len(x) for x in pfCand_pt]
import matplotlib.pyplot as plt
plt.hist(nPfCand, bins=20)
plt.title("node count")
plt.xlabel("node count")
plt.ylabel("frequency")
plt.savefig("NodeCount.png")
plt.clf()
plt.cla()

nEdges = [x*(x-1) for x in nPfCand]
plt.hist(nEdges, bins=20)
plt.title("edge count")
plt.xlabel("edge count")
plt.ylabel("frequency")
plt.savefig("EdgeCount.png")
# print(f'pfCand_eta: {pfCand_eta[a:b]}')
# print(f'pfCand_phi: {pfCand_phi[a:b]}')
# print(f'len run: {len(run)}')
# print(f'len evt: {len(evt)}')
print(f'len lumi: {len(lumi)}')
print(f'len pfCand_pt: {len(pfCand_pt)}')
print(f'len max pfCand: {ttree["pfCand_pt"].max()}')
# print(f'len pfCand_eta: {len(pfCand_eta)}')
# print(f'len pfCand_phi: {len(pfCand_phi)}')
# tau_pt = ttree['tau_pt'].array(library="np")
# pp.pprint(tau_pt)
# print(len(tau_pt))
