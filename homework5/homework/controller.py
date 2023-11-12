import pystk
import numpy as np
import datetime

# simple control 2
# vision
# deep control 2 -- policy gradient


def control(aim_point: float, current_vel: float, control_type: str = "deep", params=None) -> pystk.Action:
    from functools import partial

    if control_type == "simple":
        f = simple_control
    elif control_type == "linear":
        f = linear_control
    elif control_type == "deep":
        f = deep_control
    elif control_type == "deep_brake":
        f = deep_control_brake
    elif control_type == "deep_drift":
        f = deep_control_drift
    else:
        raise f"Unknown control_type {control_type}"

    return f(aim_point, current_vel) if params is None else f(aim_point, current_vel, params=params)


CURRENT_BEST_SIMPLE_PARAMS = dict(
    steering_scalar=0.3237, y_scalar=-0.0248, drift_angle=0.8783, accelerate_cutoff=0.8570
)


# 8/100
def simple_control(aim_point: float, current_vel: float, params: dict = CURRENT_BEST_SIMPLE_PARAMS) -> pystk.Action:
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
    accelerate_cutoff = params["accelerate_cutoff"]

    # steering the kart towards aim_point
    steering_angle = aim_point[0] * steering_scalar + aim_point[1] * y_scalar

    action.steer = np.clip(steering_angle, -1, 1)

    # enabling drift for wide turns
    if abs(steering_angle) > drift_angle:
        action.drift = True
    else:
        action.drift = False

    if abs(steering_angle) > accelerate_cutoff:
        action.brake = True
        action.acceleration = 1.0
    else:
        action.acceleration = 1.0

    return action


def simple_control_2(aim_point: float, current_vel: float, params: dict = CURRENT_BEST_SIMPLE_PARAMS) -> pystk.Action:
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
    accelerate_cutoff = params["accelerate_cutoff"]

    # steering the kart towards aim_point
    steering_angle = (
        steering_scalar * np.atan2(aim_point[0] / aim_point[1]) * np.sqrt(aim_point[0] ** 2 + aim_point[1] ** 2)
    )

    action.steer = np.clip(steering_angle, -1, 1)

    # enabling drift for wide turns
    if abs(steering_angle) > drift_angle:
        action.drift = True
    else:
        action.drift = False

    if abs(steering_angle) > accelerate_cutoff:
        action.brake = True
        action.acceleration = 1.0
    else:
        action.acceleration = 1.0

    return action


LINEAR_SIZE = (2, 3)
CURRENT_BEST_LINEAR_PARAMS = {
    "w0": -2.8160011091392554,
    "w1": 0.2512532117224574,
    "w2": 0.18879930520217497,
    "w3": 1.4292673039888695,
    "w4": -0.26967411554419424,
    "w5": 0.9915393249773421,
}


# 10/100
def linear_control(aim_point: float, current_vel: float, params: dict = CURRENT_BEST_LINEAR_PARAMS) -> pystk.Action:
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    linear = np.matrix([v for v in params.values()]).reshape(LINEAR_SIZE)

    output = linear @ np.array([*aim_point, current_vel])

    action.steer = np.clip(output.item(0), -1, 1)
    action.acceleration = 1.0 / (1.0 + np.exp(-output.item(1)))
    action.drift = output.item(0) > CURRENT_BEST_SIMPLE_PARAMS["drift_angle"]

    return action


DRIFT_LAYERS = [(6, 3), (2, 6)]
# how_far=0.5069099644349553
CURRENT_BEST_DEEP_DRIFT_PARAMS = {
    "w0": 1.1516428102031477,
    "w1": 1.1227902394251428,
    "w2": 0.7555013377402726,
    "w3": -0.3440549602041101,
    "w4": 0.03350519035320991,
    "w5": 1.112714271966624,
    "w6": 0.2586178979815187,
    "w7": -0.3928629680664677,
    "w8": -0.7636926670549435,
    "w9": 0.1475081526942162,
    "w10": -1.1358212940664671,
    "w11": -0.7588574018162755,
    "w12": 0.1030699191881122,
    "w13": -0.5613961296950909,
    "w14": -0.014646510109433191,
    "w15": -0.8441565498195834,
    "w16": 0.7108323424986188,
    "w17": -0.4906555959340215,
    "w18": 0.3552653649173329,
    "w19": -0.19922877019126442,
    "w20": 1.4815002830548696,
    "w21": 1.7729058962844295,
    "w22": -0.3714282542910141,
    "w23": -0.3461753829678067,
    "w24": -1.4373230876595957,
    "w25": -1.396944915876797,
    "w26": -0.8566980526904837,
    "w27": 0.5472473719838779,
    "w28": 1.9002478454630145,
    "w29": 0.2850339160182646,
}


def deep_control_drift(
    aim_point: float, current_vel: float, params: dict = CURRENT_BEST_DEEP_DRIFT_PARAMS
) -> pystk.Action:
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    values = [v for v in params.values()]

    linear_1 = np.matrix(values[: np.prod(DRIFT_LAYERS[0])]).reshape(DRIFT_LAYERS[0])
    output_1 = linear_1 @ np.array([*aim_point, current_vel])
    relu_1 = np.maximum(output_1, 0)

    linear_2 = np.matrix(
        values[np.prod(DRIFT_LAYERS[0]) : (np.prod(DRIFT_LAYERS[0]) + np.prod(DRIFT_LAYERS[1]))]
    ).reshape(DRIFT_LAYERS[1])
    output = linear_2 @ relu_1.T

    action.steer = np.clip(output.item(0), -1, 1)
    action.drift = output.item(1) > 0
    action.acceleration = 1.0

    return action


BRAKE_LAYERS = [(6, 3), (2, 6)]
# how_far=0.571800059921367
CURRENT_BEST_DEEP_BRAKE_PARAMS = {
    "w0": -1.9679861512734895,
    "w1": 0.7755947642736905,
    "w2": -0.5223720316484727,
    "w3": -0.13392421720538197,
    "w4": -1.4682799227292453,
    "w5": -0.2905014834767798,
    "w6": -1.2496311223007657,
    "w7": 1.3550536725487075,
    "w8": -0.09143742283106797,
    "w9": 2.170175074710688,
    "w10": 1.2141431929332498,
    "w11": -1.426846819751799,
    "w12": 0.30628687364937157,
    "w13": 0.043690226187108744,
    "w14": -0.07998873853921651,
    "w15": 0.4470514333589501,
    "w16": -1.098575444216767,
    "w17": -0.6321780017357741,
    "w18": 0.1515446037992234,
    "w19": -0.1750347011299924,
    "w20": -0.45941589714402997,
    "w21": -0.21243854052381878,
    "w22": 0.2906415857676959,
    "w23": 2.2230018607269475,
    "w24": -2.238981713828335,
    "w25": -0.07846473984977947,
    "w26": -0.8623371394363433,
    "w27": -0.46671860962956785,
    "w28": -0.32565743331846925,
    "w29": 2.1073171102855,
    "w30": 2.288388622621965,
}


def deep_control_brake(
    aim_point: float, current_vel: float, params: dict = CURRENT_BEST_DEEP_BRAKE_PARAMS
) -> pystk.Action:
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    values = [v for v in params.values()]

    linear_1 = np.matrix(values[: np.prod(BRAKE_LAYERS[0])]).reshape(BRAKE_LAYERS[0])
    output_1 = linear_1 @ np.array([*aim_point, current_vel])
    relu_1 = np.maximum(output_1, 0)

    linear_2 = np.matrix(
        values[np.prod(BRAKE_LAYERS[0]) : (np.prod(BRAKE_LAYERS[0]) + np.prod(BRAKE_LAYERS[1]))]
    ).reshape(BRAKE_LAYERS[1])
    output = linear_2 @ relu_1.T

    action.steer = np.clip(output.item(0), -1, 1)
    action.brake = output.item(1) > 0
    action.drift = output.item(0) > values[-1]
    action.acceleration = 1.0

    return action


LAYERS = [(6, 3), (3, 6)]
DEEP_SIZE = np.sum([np.prod(l) for l in LAYERS]) + 1
# how_far=0.9998399645827123 and
CURRENT_BEST_DEEP_PARAMS = {
    "w0": 0.24463741326543617,
    "w1": 0.16133385815356197,
    "w2": -0.36406581428357393,
    "w3": -1.8366201126031807,
    "w4": 0.29980400803242346,
    "w5": -1.1392804017331064,
    "w6": -0.2295716471459887,
    "w7": -0.4565519245049096,
    "w8": -1.9763834413544017,
    "w9": -0.3322936575382293,
    "w10": -2.360725006306945,
    "w11": 0.2433132645203753,
    "w12": -0.8114169896718367,
    "w13": 1.026902802257022,
    "w14": 0.10802357093288523,
    "w15": -0.40167815531479567,
    "w16": -0.028715024725307153,
    "w17": 0.808619792858487,
    "w18": 2.144811455515947,
    "w19": -1.0262777867658577,
    "w20": -2.1866959535272223,
    "w21": 1.3437839405955698,
    "w22": -0.43402987755783023,
    "w23": -0.19519811640993723,
    "w24": 0.8665190785288938,
    "w25": -0.7323037230387939,
    "w26": -1.0016766901488017,
    "w27": 0.9081075998858493,
    "w28": 0.6488800677005783,
    "w29": -0.4808397134524566,
    "w30": 0.25194112818852016,
    "w31": 0.9300698445532001,
    "w32": -1.5057113898254417,
    "w33": 0.16048697887565105,
    "w34": -1.1562035925756138,
    "w35": -0.4843793548419946,
    "w36": 0.8799677423358669,
}
# CURRENT_BEST_DEEP_PARAMS = dict([[f"w{i}", np.random.randn()] for i in range(DEEP_SIZE)])
# TUNE_MAX_PENDING_TRIALS_PG=16 python3 -m homework.controller lighthouse  -s 550 --model deep --search_alg bayes


def deep_control(aim_point: float, current_vel: float, params: dict = CURRENT_BEST_DEEP_PARAMS) -> pystk.Action:
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    values = [v for v in params.values()]

    linear_1 = np.matrix(values[: np.prod(LAYERS[0])]).reshape(LAYERS[0])
    output_1 = linear_1 @ np.array([*aim_point, current_vel])
    relu_1 = np.maximum(output_1, 0)

    linear_2 = np.matrix(values[np.prod(LAYERS[0]) : (np.prod(LAYERS[0]) + np.prod(LAYERS[1]))]).reshape(LAYERS[1])
    output = linear_2 @ relu_1.T

    action.steer = np.clip(output.item(0), -1, 1)
    action.brake = output.item(1) > 0
    action.acceleration = np.clip(output.item(2), 0.1, 1)
    action.drift = output.item(0) > values[-1]

    return action


if __name__ == "__main__":
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        if PyTux._singleton is None:
            pytux = PyTux(graphics=args.graphics)
        else:
            pytux = PyTux._singleton

        from functools import partial

        parameterized_control = partial(control, control_type=args.model)

        for t in args.track:
            steps, how_far, rescue_count = pytux.rollout(
                t,
                parameterized_control,
                max_frames=args.max_steps,
                verbose=args.verbose,
                filename=f"test_{args.model}_{'_'.join(args.track)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            print(f"{args.track}: {how_far} in {steps} steps; {rescue_count} rescues")

        PyTux._singleton = None
        pytux.close()
        del pytux

    def tune_controller(args):
        import numpy as np
        import ray
        from ray import train, tune

        context = ray.init()
        print(f"DASHBOARD URL: {context.dashboard_url}")

        def trainable(params):
            if PyTux._singleton is None:
                pytux = PyTux(graphics=args.graphics)
            else:
                pytux = PyTux._singleton

            from functools import partial

            parameterized_control = partial(control, params=params, control_type=args.model)

            # TODO: how to track multiple tracks
            for t in args.track:
                steps, how_far, rescue_count = pytux.rollout(
                    t,
                    parameterized_control,
                    max_frames=args.max_steps,
                    verbose=args.verbose,
                    train=train,
                    break_on_rescue=True,
                )

            PyTux._singleton = None
            pytux.close()
            del pytux
            return {"steps": steps, "how_far": how_far, "rescue_count": rescue_count}

        import os

        # Get the current working directory
        cwd = os.getcwd()

        # Create the absolute path to the log directory
        log_dir = os.path.join(cwd, "log")

        epsilon = 1e-1

        if args.model == "simple":
            best_points = [CURRENT_BEST_SIMPLE_PARAMS]
            param_space = {
                "steering_scalar": tune.uniform(0.0, 3.0),
                "y_scalar": tune.uniform(0.0, 1.0),
                "drift_angle": tune.uniform(0.5, 1.0),
                "accelerate_cutoff": tune.uniform(0.5, 1.0),
            }
        elif args.model == "linear":
            best_points = [CURRENT_BEST_LINEAR_PARAMS]
            size = np.prod(LINEAR_SIZE)
            param_space = dict([[f"w{i}", tune.randn()] for i in range(size)])

            if args.search_alg == "random":
                param_space = dict([[f"w{i}", tune.randn()] for i in range(size)])
            else:
                param_space = dict(
                    [
                        [
                            f"w{i}",
                            tune.uniform(
                                CURRENT_BEST_LINEAR_PARAMS[f"w{i}"] - epsilon,
                                CURRENT_BEST_LINEAR_PARAMS[f"w{i}"] + epsilon,
                            ),
                        ]
                        for i in range(size)
                    ]
                )
        elif args.model == "deep":
            best_points = [CURRENT_BEST_DEEP_PARAMS]

            if args.search_alg == "random":
                param_space = dict([[f"w{i}", tune.randn()] for i in range(DEEP_SIZE)])
            else:
                param_space = dict(
                    [
                        [
                            f"w{i}",
                            tune.uniform(
                                CURRENT_BEST_DEEP_PARAMS[f"w{i}"] - epsilon,
                                CURRENT_BEST_DEEP_PARAMS[f"w{i}"] + epsilon,
                            ),
                        ]
                        for i in range(DEEP_SIZE)
                    ]
                )

        if args.search_alg == "pdb":
            scheduler = tune.schedulers.PopulationBasedTraining(
                time_attr="steps",
                perturbation_interval=1,
                hyperparam_mutations={
                    "steering_scalar": tune.uniform(0.0, 5.0),
                    "drift_angle": tune.uniform(0.0, 1.0),
                },
            )
            search_alg = None
        elif args.search_alg == "random":
            scheduler = tune.schedulers.ASHAScheduler(time_attr="steps", max_t=args.max_steps, grace_period=50)

            from ray.tune.search.basic_variant import BasicVariantGenerator

            search_alg = BasicVariantGenerator(points_to_evaluate=best_points)
        elif args.search_alg == "bayes":
            scheduler = tune.schedulers.ASHAScheduler(time_attr="steps", max_t=args.max_steps, grace_period=50)

            from ray.tune.search.bayesopt import BayesOptSearch

            search_alg = BayesOptSearch(metric="how_far", mode="max", points_to_evaluate=best_points)
        elif args.search_alg == "hyperopt":
            scheduler = tune.schedulers.ASHAScheduler(time_attr="steps", max_t=args.max_steps, grace_period=50)

            from ray.tune.search.hyperopt import HyperOptSearch

            search_alg = HyperOptSearch(
                metric="how_far", mode="max", n_initial_points=10, points_to_evaluate=best_points
            )
        elif args.search_alg == "hyperband":
            scheduler = tune.schedulers.HyperBandScheduler(time_attr="steps", max_t=args.max_steps)

            from ray.tune.search.hyperopt import HyperOptSearch

            search_alg = HyperOptSearch(
                metric="how_far", mode="max", n_initial_points=10, points_to_evaluate=best_points
            )
        elif args.search_alg == "bohb":
            scheduler = tune.schedulers.HyperBandForBOHB(time_attr="steps", max_t=args.max_steps)

            from ray.tune.search.bohb import TuneBOHB

            search_alg = TuneBOHB(metric="how_far", mode="max", points_to_evaluate=best_points)
        elif args.search_alg == "ax":
            scheduler = tune.schedulers.ASHAScheduler(time_attr="steps", max_t=args.max_steps, grace_period=50)

            from ray.tune.search.ax import AxSearch

            search_alg = AxSearch(metric="how_far", mode="max", points_to_evaluate=best_points)

        tuner = tune.Tuner(
            tune.with_resources(trainable, {"cpu": 1, "gpu": 0}),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                mode="max",
                metric="how_far",
                num_samples=200,
                reuse_actors=False,
            ),
            run_config=train.RunConfig(
                local_dir=log_dir,
                name=f"tune_controller_{args.model}_{args.search_alg}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ),
        )

        results_grid = tuner.fit()
        print(results_grid.get_best_result())

    parser = ArgumentParser()
    parser.add_argument("track", nargs="+")
    parser.add_argument("-s", "--max_steps", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--test_run", action="store_true")
    parser.add_argument("-g", "--graphics", action="store_true")
    parser.add_argument("--search_alg", default="random")
    parser.add_argument("--model", default="simple")
    args = parser.parse_args()
    if args.test_run:
        test_controller(args)
    else:
        tune_controller(args)
