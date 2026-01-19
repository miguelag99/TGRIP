VEL_THRESHOLD = 0.5  # m/s

# Conditions used for semantic maps are filled in dataloader using CLIP embeddings

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

MAP_LAYERS = {
    "background": {"text": "Empty background with no objects", "idx": 0},
    "drivable_area": {
        "text": "A flat asphalt road surface where vehicles can travel",
        "idx": 1,
    },
    "road_divider": {
        "text": "A white or yellow painted line separating traffic lanes on a street",
        "idx": 2,
    },
    "ped_crossing": {
        "text": "A zebra crossing with white stripes for pedestrians to cross the road",
        "idx": 3,
    },
    "stop_line": {
        "text": "A thick white solid line on the road where cars must stop at an intersection",
        "idx": 4,
    },
    "walkway": {
        "text": "A paved sidewalk or pedestrian path next to the road",
        "idx": 5,
    },
}