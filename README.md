# Proof: Maximum Reachability Problem (MRP) is NP-hard

## Lemma 3.1: MRP is NP-hard

We prove this by reduction from the Maximum Coverage Problem (MCP), which is known to be NP-hard. 

In MCP, given:
- A universe **U** of elements
- A collection of subsets **S₁, S₂, ..., Sₙ** of **U**
- An integer **K**

The goal is to select at most **K** subsets so that the number of covered elements is maximized.

### Construction for Reduction:
1. **Mapping**:
   - Construct a directed graph **G = (V, E)** where:
     - Each element in **U** is represented as a vertex in **V**
     - For each subset **Sᵢ** in MCP, create a vertex **vᵢ** in **V**
     - Add a directed edge from vertex **vᵢ** to vertex **vⱼ** if element **j** in **U** belongs to subset **Sᵢ**
   - The objective in the MRP is to select **K** vertices **V'** ⊆ **V** such that the number of vertices reachable from **V'** is maximized, which correlates directly to covering elements in MCP.

### Correctness:
1. **Completeness**:
   - If selecting subsets **Sᵢ₁, Sᵢ₂, ..., Sᵢₖ** covers the maximum number of elements in **U** in MCP, then selecting vertices **vᵢ₁, vᵢ₂, ..., vᵢₖ** in **G** maximizes the reachability to other vertices.
2. **Soundness**:
   - Conversely, if a set of vertices **V'** in **G** maximizes the number of reachable vertices, then the corresponding subsets in MCP would cover a maximum number of elements in **U**.

Therefore, solving the MRP is at least as hard as solving MCP. Since MCP is NP-hard, by polynomial-time reduction, the **MRP is also NP-hard**.

---

### Alternate Framing: Budgeted Maximum Set Cover Problem

The Budgeted Maximum Set Cover Problem (BMSCP) is defined as follows:
- A universe set **U** comprises **n** elements, each assigned a utility value.
- A collection **S** contains several subsets of **U**, each with an associated cost.
- Given a budget **B**, the problem seeks to identify the smallest subset of **S** that maximizes the total utilities of the covered elements while keeping the total cost within **B**.

### Relation to Maximum Reachability Problem:
- Treat the vertex set **V** as the universe set **U**.
- View each node as a set comprising all its reachable nodes. These sets collectively represent **S**.
- Assume:
  - Each node has a utility of 1.
  - The cost to acquire each set in **S** is also 1.
- The goal is to select fewer than **B** nodes such that the union of their expanded sets yields the maximum total utility.

Consequently, the Maximum Reachability Problem (MRP) can be framed as a **Budgeted Maximum Set Cover Problem**, which is also **NP-hard**.





[11]Alberto Caprara, Paolo Toth, and Matteo Fischetti. 2000. Algorithms for the set
covering problem. Annals of Operations Research 98, 1 (2000), 353–371.
[19]Samir Khuller, Anna Moss, and Joseph Seffi Naor. 1999. The budgeted maximum
coverage problem. Information processing letters 70, 1 (1999), 39–45




