# Protein Crystallization Modeling Pipeline
ML model to predict protein crystallization process
This system allows you to:
<BR>✅ Predict outcomes (resolution, space group) for given conditions (forward prediction)
<BR>🔄 Generate new conditions to achieve a desired outcome (inverse design)

                      +-----------------------------+
                      |      Protein Sequence       |
                      +-----------------------------+
                                   |
             +---------------------+---------------------+
             |                                           |
    +--------v---------+                    +------------v----------+
    |  Forward Model   |                    |    Inverse Generator  |
    | (predict result) |                    |  (generate conditions)|
    +------------------+                    +------------------------+
             |                                           |
     Outcome prediction                      Generated conditions for target
             |                                           |
       Model tuning                            Run forward model for scoring
             |                                           |
      +------v------+                           +--------v--------+
      |  Lab/Screen |<------------------------->| Loop + Refinement|
      +-------------+                           +------------------+

✅ 1. Forward Model (Predict Result from Sequence + Conditions)
Model Inputs:
ESM embedding from sequence

Crystallization features:
 -pH, temperature
 -TF-IDF or embedding of details string

Model Target:
Regression: resolution (float)

Classification: diffraction quality (high/medium/low)



