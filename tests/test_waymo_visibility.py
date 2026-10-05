"""Waymo has no rear camera: a car overtaken by the ego leaves every camera FOV while the model, which
saw it in the input frames, keeps predicting it. Such observed objects must stay valid (supervised and
scored) in the future BEV frames; objects never seen by the cameras must stay ignored."""

from tgrip.data.dataset.waymo_temporal import NOT_VISIBLE, VISIBLE, WaymoDB

MIN_VIS = 2


def _db():
    db = WaymoDB(dataroot="", img_height=886)
    anns = {
        # Overtaken car: visible at t0, out of every camera FOV at t+1.
        ("t0", "overtaken"): VISIBLE,
        ("t1", "overtaken"): NOT_VISIBLE,
        # Car behind the ego, never in any camera.
        ("t0", "behind"): NOT_VISIBLE,
        ("t1", "behind"): NOT_VISIBLE,
    }
    for (sample, inst), vis in anns.items():
        db._tables["sample_annotation"][f"{sample}_{inst}"] = {
            "instance_token": inst,
            "visibility_token": vis,
        }
    samples = {s: {"anns": [f"{s}_overtaken", f"{s}_behind"]} for s in ["t0", "t1"]}
    return db, samples


def test_observed_object_stays_valid_after_leaving_fov():
    db, samples = _db()
    db.observed_instances = db.visible_instances([samples["t0"]], MIN_VIS)

    assert db.get("sample_annotation", "t1_overtaken")["visibility_token"] >= MIN_VIS
    # Never observed: the model has no evidence of it, so it stays out of the loss and metrics.
    assert db.get("sample_annotation", "t1_behind")["visibility_token"] < MIN_VIS


def test_observed_instances_do_not_leak_into_stored_annotations():
    db, samples = _db()
    db.observed_instances = db.visible_instances([samples["t0"]], MIN_VIS)
    db.get("sample_annotation", "t1_overtaken")
    db.observed_instances = set()

    # Another sample whose input frames did not see the car must get its own-frame visibility.
    assert db.get("sample_annotation", "t1_overtaken")["visibility_token"] == NOT_VISIBLE
