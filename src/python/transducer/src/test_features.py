from features import Features
import sys

def read_data(file_in):
    data = []
    with open(file_in, 'rb') as f:
        for line in f:
            line = line.strip()
            upper, lower = line.split("\t")
            data.append((upper, lower))
    return data
            
Sigma, Sigma_inv = {}, {}
data = read_data(sys.argv[1])
for upper, lower in data:
    print upper, lower
    for i, c in enumerate(set(list(upper))):
        if c not in Sigma:
            Sigma_inv[len(Sigma)] = c
            Sigma[c] = len(Sigma)

    for i, c in enumerate(set(list(lower))):
        if c not in Sigma:
            Sigma_inv[len(Sigma)] = c
            Sigma[c] = len(Sigma)
            
        
        
features = Features(Sigma, Sigma_inv)
for upper, lower in data:
    #print upper, lower, len(features.features)
    features.extract(upper, URC=0, ULC=0,create=True)
    break

print len(features.features)

print features.num_extracted


for k, v in  features.features.items():
    print k, v
#print features._right_context(2, "hello", 4)
#print features._left_context(2, "helloword", 7)

