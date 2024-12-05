# Search of an FD Graph

## 4.2.1 Pruning
Our functional dependency method is based on a mutual information/entropy approach to search for functional dependency. This method focuses on calculating the mutual information of column pairs in the dataset, and we set a mutual information threshold of 0.975. If the mutual information of two columns is greater than 0.975, they are considered functionally dependent, as described in reference [18].

We also follow the same pruning steps as in [18]:
- First, we prune columns that have only one or two distinct values.
- For columns with a large number of distinct values, we use a constant ε3 ∈ [0,1] for pruning. We set ε3 to 0.95.

### Algorithm 1: Pruning before FD Search

```
Get total number of rows |R| in the dataset
for each column Ci in the dataset do
    Get distinct values for Ci from the dictionary
    if |Ci| < 3 then
        Prune Ci
        continue
    end if
    if |Ci| > (1 - ε3) · |R| then
        Prune Ci
        continue
    end if
end for
```

## 4.2.2 Pruning FD Search
After pruning, we perform the original algorithm from [1] to find functional dependencies, except that we do not use the Chao Shen entropy as in [1]. Instead, we randomly sample 10,000 rows from the original dataset and calculate the exact mutual information.

### Algorithm 2: FD Search

```
for all possible pairs (Ci, Cj) in the dataset sample (after pruning) do
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
    if (ωc_i,Cj ≥ X) or (ωc_j,Ci ≥ X) then
        Pair (Ci, Cj) is added to functional dependency
    end if
end for
```

## References

[1] Marcus Paradies, Christian Lemke, Hasso Plattner, Wolfgang Lehner, Kai-Uwe Sattler, Alexander Zeier, and Jens Krueger. 2010. How to juggle columns. Proceedings of the Fourteenth International Database Engineering & Applications Symposium on - IDEAS '10 (2010). [https://doi.org/10.1145/1](https://doi.org/10.1145/1)


