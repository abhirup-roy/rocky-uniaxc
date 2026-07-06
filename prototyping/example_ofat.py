import rocky_uniaxc as uniaxc

bb_sched = uniaxc.RockyScheduler.bb_gpu()

ofat_values = {
    "parameters": ["cor_pp", "surf_en_pp"],
    "test_range": [(0.1, 0.7), (0, 100)],
    "hold_values": ["m", "l"],
}

uniaxc.launch_ofat(
    sweep_name="test_ofat",
    scheduler=bb_sched,
    ofat_values=ofat_values,
    n_points=5,
    json_path="json/ofat_base.json",
    autolaunch=False,
    target="gpu",
    backend="rocky_prepost",
)
