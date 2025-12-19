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
    "background": {"text": "Empty background with no objects", "idx": 0},
    "vehicle.bicycle": {"text": "A photo of a bicycle", "idx": 1},
    "vehicle.bus.bendy": {"text": "A photo of a bus", "idx": 2},
    "vehicle.bus.rigid": {"text": "A photo of a bus", "idx": 3},
    "vehicle.car": {"text": "A photo of a car", "idx": 4},
    "vehicle.construction": {"text": "A photo of a construction vehicle", "idx": 5},
    "vehicle.emergency.ambulance": {"text": "A photo of an ambulance", "idx": 6},
    "vehicle.emergency.police": {"text": "A photo of a police car", "idx": 7},
    "vehicle.motorcycle": {"text": "A photo of a motorcycle", "idx": 8},
    "vehicle.trailer": {"text": "A photo of a trailer", "idx": 9},
    "vehicle.truck": {"text": "A photo of a truck", "idx": 10},
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