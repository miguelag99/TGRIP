VEL_THRESHOLD = 0.5  # m/s

# Conditions used for semantic maps are filled in dataloader using CLIP embeddings
VELOCITY_CONDITIONS = {
    "background": {"text": "Background", "idx": 0},
    "vehicle.stopped": {"text": "Stopped", "idx": 1},
    "vehicle.parked": {"text": "Stopped", "idx": 2},
    "cycle.without_rider": {"text": "Stopped", "idx": 3},
    "vehicle.moving": {"text": "Moving", "idx": 4},
    "cycle.with_rider": {"text": "Moving", "idx": 5},
}

CLASS_CONDITIONS = {
    "background": {"text": "Background", "idx": 0},
    "vehicle.bicycle": {"text": "Bicycle", "idx": 1},
    "vehicle.bus.bendy": {"text": "Bus", "idx": 2},
    "vehicle.bus.rigid": {"text": "Bus", "idx": 3},
    "vehicle.car": {"text": "Car", "idx": 4},
    "vehicle.construction": {"text": "Construction Vehicle", "idx": 5},
    "vehicle.emergency.ambulance": {"text": "Ambulance", "idx": 6},
    "vehicle.emergency.police": {"text": "Police Car", "idx": 7},
    "vehicle.motorcycle": {"text": "Motorcycle", "idx": 8},
    "vehicle.trailer": {"text": "Trailer", "idx": 9},
    "vehicle.truck": {"text": "Truck", "idx": 10},
}

POSITIONAL_CONDITIONS= {
    'background': {"text": "Background", "idx": 0},
    'front': {"text": "Front", "idx": 1},
    'front_left': {"text": "Front Left", "idx": 2},
    'front_right': {"text": "Front Right", "idx": 3},
    'back_left': {"text": "Back Left", "idx": 4},
    'back_right': {"text": "Back Right", "idx": 5},
    'back': {"text": "Back", "idx": 6},
}