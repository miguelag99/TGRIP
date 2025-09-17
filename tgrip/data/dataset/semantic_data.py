VEL_THRESHOLD = 0.5  # m/s
SCENE_TEXT_CONDITIONS = [
    # No conditions
    # {
    #     "text_condition": "All vehicles",
    #     "keyword": "all",
    # },
    # Dynamic tags conditions
    {
        "text_condition": "Moving vehicle",
        "filter_by": "attribute_tokens",
        "keyword": "moving",
        "values": ["vehicle.moving", "cycle.with_rider"]
    },
    {
        "text_condition": "Stopped vehicle",
        "filter_by": "attribute_tokens",
        "keyword": "stopped",
        "values": ["vehicle.stopped", "vehicle.parked", "cycle.without_rider"]
    },
    # {
    #     "text_condition": "Parked vehicle",
    #     "filter_by": "attribute_tokens",
    #     "keyword": "parked",
    #     "values": ["vehicle.parked", "cycle.without_rider"]
    # },
    # Class conditions
    # {
    #     "text_condition": "Car",
    #     "filter_by": "category_name",
    #     "keyword": "category",
    #     "values": "vehicle.car"
    # },
    # {
    #     "text_condition": "Bicycle",
    #     "filter_by": "category_name",
    #     "keyword": "category",
    #     "values": "vehicle.bicycle"
    # },
    # {
    #     "text_condition": "Truck",
    #     "filter_by": "category_name",
    #     "keyword": "category",
    #     "values": "vehicle.truck"
    # },
    # {
    #     "text_condition": "Bus",
    #     "filter_by": "category_name",
    #     "keyword": "category",
    #     "values": "vehicle.bus"
    # },
    # {
    #     "text_condition": "Trailer",
    #     "filter_by": "category_name",
    #     "keyword": "category",
    #     "values": "vehicle.trailer"
    # }
]

# 
VELOCITY_CONDITIONS = {
    "vehicle.stopped": {"text": "Stopped vehicle"},
    "vehicle.parked": {"text": "Stopped vehicle"},
    "cycle.without_rider": {"text": "Stopped vehicle"},
    "vehicle.moving": {"text": "Moving vehicle"},
    "cycle.with_rider": {"text": "Moving vehicle"},
}

# Created inside dataloader
POSITIONAL_CONDITIONS= {
    'front': None,
    'front_left': None,
    'front_right': None,
    'back_left': None,
    'back_right': None,
    'back': None,
}