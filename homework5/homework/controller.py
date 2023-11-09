import pystk
import numpy as np


CURRENT_BEST_SIMPLE_PARAMS = dict(steering_scalar=0.248227, y_scalar=0.0, drift_angle=0.843254)

def control(aim_point: float, current_vel: float, params: dict = CURRENT_BEST_SIMPLE_PARAMS) -> pystk.Action:
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    action = pystk.Action()
    steering_scalar = params["steering_scalar"]
    y_scalar = params["y_scalar"]
    drift_angle = params["drift_angle"]

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    action.acceleration = 1.0

    # steering the kart towards aim_point
    steering_angle = aim_point[0] * steering_scalar + aim_point[1] * y_scalar

    action.steer = np.clip(steering_angle, -1, 1)

    # enabling drift for wide turns
    if abs(steering_angle) > drift_angle:
        action.drift = True
    else:
        action.drift = False

    return action


if __name__ == "__main__":
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()

        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=args.max_steps, verbose=args.verbose)
            print(f"{args.track}: {how_far} in {steps} steps")

        pytux.close()

    def tune_controller(args):
        import numpy as np
        from ray import train, tune

        def trainable(params):
            pytux = PyTux()
            from functools import partial

            parameterized_control = partial(control, params=params)

            # TODO: how to track multiple tracks
            for t in args.track:
                steps, how_far = pytux.rollout(
                    t,
                    parameterized_control,
                    max_frames=args.max_steps,
                    verbose=args.verbose,
                    train=train,
                    break_on_rescue=True,
                )

            pytux.close()
            return {"steps": steps, "how_far": how_far}

        trainable_with_resources = tune.with_resources(trainable, {"cpu": 16})

        import os

        # Get the current working directory
        cwd = os.getcwd()

        # Create the absolute path to the log directory
        log_dir = os.path.join(cwd, "log")

        if args.pbt:
            scheduler = tune.schedulers.PopulationBasedTraining(
                time_attr="steps",
                perturbation_interval=1,
                hyperparam_mutations={
                    "steering_scalar": tune.uniform(0.0, 5.0),
                    "drift_angle": tune.uniform(0.0, 1.0),
                },
            )
            search_alg = None
        else:
            scheduler = tune.schedulers.ASHAScheduler(
                time_attr="steps", max_t=args.max_steps, grace_period=100
            )
            # from ray.tune.search.bayesopt import BayesOptSearch

            # search_alg = BayesOptSearch(
            #     metric="how_far", mode="max", random_search_steps=10, points_to_evaluate=[CURRENT_BEST_SIMPLE_PARAMS]
            # )

            # from ray.tune.search.hyperopt import HyperOptSearch

            # search_alg = tune.search.hyperopt.HyperOptSearch(metric="how_far", mode="max", n_initial_points=10, points_to_evaluate=[CURRENT_BEST_SIMPLE_PARAMS])

        tuner = tune.Tuner(
            trainable_with_resources,
            param_space={
                "steering_scalar": tune.randn(CURRENT_BEST_SIMPLE_PARAMS["steering_scalar"], 0.1),
                "y_scalar": tune.randn(CURRENT_BEST_SIMPLE_PARAMS["y_scalar"], 0.1),
                "drift_angle": tune.randn(CURRENT_BEST_SIMPLE_PARAMS["drift_angle"], 0.1),
            },
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                # search_alg=search_alg,
                max_concurrent_trials=16,
                mode="max",
                metric="how_far",
                num_samples=100,
                reuse_actors=False,
            ),
            run_config=train.RunConfig(storage_path=f"file://{log_dir}", name="tune_controller"),
        )

        results_grid = tuner.fit()
        print(results_grid.get_best_result())

    parser = ArgumentParser()
    parser.add_argument("track", nargs="+")
    parser.add_argument("-s", "--max_steps", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--test_run", action="store_true")
    parser.add_argument("--pbt", action="store_true")
    args = parser.parse_args()
    if args.test_run:
        test_controller(args)
    else:
        tune_controller(args)
