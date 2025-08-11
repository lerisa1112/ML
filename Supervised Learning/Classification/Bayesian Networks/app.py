# Example A: manual Bayesian Network with pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1) Define DAG structure
model = BayesianNetwork([
    ('Cloudy', 'Sprinkler'),
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'WetGrass'),
    ('Rain', 'WetGrass'),
])

# 2) Define CPDs
# Cloudy: P(Cloudy)
cpd_cloudy = TabularCPD(variable='Cloudy', variable_card=2,
                        values=[[0.5],    # Cloudy = 0 (False)
                                [0.5]])   # Cloudy = 1 (True)

# Sprinkler: P(Sprinkler | Cloudy)
# Order of evidence variables defaults to the order listed in model edges: here 'Cloudy'
# Rows correspond to Sprinkler=0, Sprinkler=1
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.5, 0.9],   # Sprinkler=False when Cloudy=False,Cloudy=True
                                   [0.5, 0.1]],
                           evidence=['Cloudy'],
                           evidence_card=[2])

# Rain: P(Rain | Cloudy)
cpd_rain = TabularCPD(variable='Rain', variable_card=2,
                      values=[[0.8, 0.2],   # Rain=False given Cloudy=False/True
                              [0.2, 0.8]],
                      evidence=['Cloudy'],
                      evidence_card=[2])

# WetGrass: P(WetGrass | Sprinkler, Rain)
# Evidence order: ['Sprinkler','Rain'] -> evidence_card [2,2]
# Rows: WetGrass=False, WetGrass=True
cpd_wetgrass = TabularCPD(variable='WetGrass', variable_card=2,
                          values=[
                              # Sprinkler=0, Rain=0 | Sprinkler=0, Rain=1 | Sprinkler=1, Rain=0 | Sprinkler=1, Rain=1
                              [1.0,           0.1,            0.1,            0.01],  # WetGrass=False
                              [0.0,           0.9,            0.9,            0.99]   # WetGrass=True
                          ],
                          evidence=['Sprinkler', 'Rain'],
                          evidence_card=[2, 2])

# 3) Add CPDs to model and validate
model.add_cpds(cpd_cloudy, cpd_sprinkler, cpd_rain, cpd_wetgrass)
assert model.check_model(), "Model is invalid (CPDs inconsistent)"

# 4) Inference
infer = VariableElimination(model)

# Example queries:
# a) P(Rain) (marginal)
print("P(Rain):")
print(infer.query(variables=['Rain']))

# b) P(Rain | WetGrass=True)
print("\nP(Rain | WetGrass=True):")
print(infer.query(variables=['Rain'], evidence={'WetGrass': 1}))

# c) P(Cloudy | WetGrass=True)
print("\nP(Cloudy | WetGrass=True):")
print(infer.query(variables=['Cloudy'], evidence={'WetGrass': 1}))
