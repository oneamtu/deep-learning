import pystk
import numpy as np

def control(aim_point, current_vel, steering_scalar=1., drift_angle=.5):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    action.acceleration = 1.0

    # steering the kart towards aim_point
    steering_angle = aim_point[0] * steering_scalar

    action.steer = np.clip(steering_angle, -1, 1)

    drift_angle = 0.5

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
        import numpy as np
        from ray import train, tune

        def trainable(params):
            pytux = PyTux()
            from functools import partial

            parameterized_control = partial(control, steering_scalar=params["steering_scalar"], drift_angle=params["drift_angle"])

            # TODO: how to track multiple tracks
            for t in args.track:
                steps, how_far = pytux.rollout(t, parameterized_control, max_frames=1000, verbose=args.verbose)

            pytux.close()
            # TODO: log this to tune as we go
            return {"steps": steps, "how_far": how_far}

        import os

        # Get the current working directory
        cwd = os.getcwd()

        # Create the absolute path to the log directory
        log_dir = os.path.join(cwd, 'log')

        scheduler = tune.schedulers.ASHAScheduler(
            time_attr='steps',
            max_t=100,
            grace_period=1,
            reduction_factor=2)

        tuner = tune.Tuner(trainable,
            param_space={"steering_scalar": tune.uniform(0., 5.), "drift_angle": tune.uniform(0., 1.)},
            tune_config=tune.TuneConfig(mode="max", metric="how_far", num_samples=10, scheduler=scheduler),
            run_config=train.RunConfig(storage_path=f"file://{log_dir}", name="tune_controller"))

        results_grid = tuner.fit()
        print(results_grid.get_best_result())

    parser = ArgumentParser()
    parser.add_argument("track", nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    test_controller(args)
