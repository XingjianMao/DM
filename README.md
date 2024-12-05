# Search of an FD Graph

## 4.2.1 Pruning
Our functional dependency method is based on a mutual information/entropy approach to search for functional dependency. This method focuses on calculating the mutual information of column pairs in the dataset, and we set a mutual information threshold of 0.975. If the mutual dependence information of two columns is bigger than 0.975, then it is considered functional dependent, which is the same in reference [18]. We also follow the same prune step: first, we prune the columns that have the number of distinct values of one or two. And the ones with several distinct values that are too big, we use the same method as [18], set a constant 𝜖3 ∈ [0,1] for pruning. We set 𝜖3 to 0.95.

### Algorithm 1: Pruning before FD Search

```plaintext
Get total number of rows |R| in the dataset
for each column Ci in the dataset do
    Get distinct values for Ci from the dictionary
    if |Ci| < 3 then
        Prune Ci
        continue
    end if
    if |Ci| > (1 - 𝜖3) · |R| then
        Prune Ci
        continue
    end if
end for


4.2.2 FD Search
After pruning, we perform the original algorithm of [18] to find functional dependency, except that we don’t use the Chao Shen entropy as [18] did. We randomly sample 10000 rows from the original dataset and calculate the exact mutual information.

Algorithm 2: FD Search
plaintext
for all possible pairs (Ci, Cj) in the dataset sample (after prune) do
    for i = 0 to n do
        Get val1 from Ci
        Get val2 from Cj
        Update freqCi with val1
        Update freqCj with val2
        Update freqCi,Cj with val1 and val2
        if |freqCi,Cj| > (1 + X) · max{|Ci|, |Cj|} then
            break
        end if
    end for
    hc_i = ENTROPY(freqCi)
    hc_j = ENTROPY(freqCj)
    hP_c = COENTROPY(freqCi,Cj)
    ωc_i,Cj = (hc_i + hc_j - hP_c)/hc_j
    ωc_j,Ci = (hc_i + hc_j - hP_c)/hc_i
    if (ωc_i,Cj ≥ X)or(ωc_j,Ci ≥ X) then
        Pair (Ci, Cj) is added to functional dependency
    end if
end for
