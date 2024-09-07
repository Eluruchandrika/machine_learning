import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
price_change = ctrl.Antecedent(np.arange(-10, 11, 1), 'price_change')
volume_change = ctrl.Antecedent(np.arange(-10, 11, 1), 'volume_change')
price_movement = ctrl.Consequent(np.arange(-10, 11, 1), 'price_movement')

# Define fuzzy membership functions
price_change['decrease'] = fuzz.trimf(price_change.universe, [-10, -10, 0])
price_change['steady'] = fuzz.trimf(price_change.universe, [-10, 0, 10])
price_change['increase'] = fuzz.trimf(price_change.universe, [0, 10, 10])

volume_change['low'] = fuzz.trimf(volume_change.universe, [-10, -10, 0])
volume_change['medium'] = fuzz.trimf(volume_change.universe, [-10, 0, 10])
volume_change['high'] = fuzz.trimf(volume_change.universe, [0, 10, 10])

price_movement['decrease'] = fuzz.trimf(price_movement.universe, [-10, -10, 0])
price_movement['steady'] = fuzz.trimf(price_movement.universe, [-10, 0, 10])
price_movement['increase'] = fuzz.trimf(price_movement.universe, [0, 10, 10])

# Define fuzzy rules
rule1 = ctrl.Rule(price_change['increase'] & volume_change['high'], price_movement['increase'])
rule2 = ctrl.Rule(price_change['decrease'] & volume_change['low'], price_movement['decrease'])
rule3 = ctrl.Rule(price_change['steady'] & volume_change['medium'], price_movement['steady'])

# Create a control system
price_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
price_sim = ctrl.ControlSystemSimulation(price_ctrl)

# Input data
price_sim.input['price_change'] = 6
price_sim.input['volume_change'] = 8

# Compute the output
price_sim.compute()

# Output the predicted price movement
print(f"Predicted Price Movement: {price_sim.output['price_movement']:.2f}")
