import pystk
import numpy as np
import datetime

from .planner import load_model

# vision
# update stats
# nitro?
# simple control 2
# deep control 2 -- policy gradient
# DQNs
# tune callback


def control(aim_point: float, current_vel: float, control_type: str = "simple", params=None) -> pystk.Action:
    from functools import partial

    if control_type == "simple":
        f = simple_control
    elif control_type == "simple_2":
        f = simple_control_2
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


CURRENT_BEST_SIMPLE_PARAMS = {
    "steering_scalar": 2.5232161324474416,
    "drift_angle": 0.9220483178471008,
    "accelerate_cutoff": 0.5045246985540843,
}


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
    y_scalar = 0.
    drift_angle = params["drift_angle"]
    accelerate_cutoff = params["accelerate_cutoff"]

    # steering the kart towards aim_point
    steering_angle = aim_point[0] * steering_scalar

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


SIMPLE_SIZE = 4
CURRENT_BEST_SIMPLE_2_PARAMS = {
    "aim_steering": 0.21101163039466375,
    "vel_steering": 0.521437424136858,
    "drift_cutoff": 0.5474835823774674,
    "brake_cutoff": 0.49755886788365955,
    "vel_max": 19.620181982924144,
}


def simple_control_2(aim_point: float, current_vel: float, params: dict = CURRENT_BEST_SIMPLE_2_PARAMS) -> pystk.Action:
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    action = pystk.Action()
    aim_steering = params["aim_steering"]
    vel_steering = params["vel_steering"]
    drift_cutoff = params["drift_cutoff"]
    brake_cutoff = params["brake_cutoff"]
    vel_max = params["vel_max"]

    current_vel_n = current_vel / 30.0

    # steering the kart towards aim_point
    steering_angle = aim_point[0] * aim_steering + np.sign(aim_point[0]) * vel_steering * current_vel_n

    steer_abs = abs(steering_angle)

    action.steer = np.clip(steering_angle, -1, 1)
    action.drift = steer_abs > drift_cutoff
    action.brake = steer_abs > brake_cutoff
    # action.acceleration = 1.0 - (current_vel_n * vel_acceleration + steer_abs * steer_acceleration)

    if current_vel < vel_max:
        action.acceleration = 1.0
    else:
        action.acceleration = 0.0

    action.nitro = steer_abs < 0.1

    return action


LINEAR_SIZE = (2, 3)
CURRENT_BEST_LINEAR_PARAMS = {
    "w0": 1.1689493979701004,
    "w1": -0.32592878018870886,
    "w2": -0.1681750191262573,
    "w3": -1.0557620970444317,
    "w4": -0.028418626212363848,
    "w5": -0.703920327286141,
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
# how_far=0.9903986373370269 and
CURRENT_BEST_DEEP_PARAMS = {
    "w0": 1.5139399163489617,
    "w1": 0.6232643605625922,
    "w2": 0.3325669065339777,
    "w3": -0.47447597495357063,
    "w4": 0.14610372867942328,
    "w5": -1.158369742120721,
    "w6": 0.8460383514898953,
    "w7": -0.5981810141163935,
    "w8": 0.3187762338568961,
    "w9": 0.38787357141421375,
    "w10": -0.7029404600107604,
    "w11": 2.1733650778946223,
    "w12": -0.46765052851835615,
    "w13": 1.112288765273333,
    "w14": -0.04828526741304289,
    "w15": 0.32780195331477024,
    "w16": -0.98475622685582,
    "w17": -0.6426524214633156,
    "w18": 1.2717666611679634,
    "w19": 0.8815735767768772,
    "w20": -0.4046396203917234,
    "w21": 0.030931108529260704,
    "w22": -0.5784019823386102,
    "w23": -0.1439124855187288,
    "w24": -0.727572625218137,
    "w25": 1.9286879181386072,
    "w26": 1.8387263591764167,
    "w27": -1.375401806887801,
    "w28": 0.8839426882176716,
    "w29": 0.562257961678059,
    "w30": -0.5711142106590732,
    "w31": -0.7775833749653532,
    "w32": 0.9809670415754805,
    "w33": 0.09996655057279193,
    "w34": -0.5235771770731711,
    "w35": -1.4462690367221314,
    "w36": 1.3435816512816872,
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

        if args.planner:
            planner = load_model().eval()
        else:
            planner = None

        for t in args.track:
            steps, how_far, rescue_count = pytux.rollout(
                t,
                parameterized_control,
                max_frames=args.max_steps,
                verbose=args.verbose,
                planner=planner,
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

            if args.planner:
                planner = load_model().eval()
            else:
                planner = None

            def train_callback(steps, how_far, rescue_count):
                if steps % 10 == 0:
                    train.report(
                        {
                            "steps": steps + total_steps,
                            "how_far": how_far + total_how_far,
                            "rescue_count": rescue_count + total_rescue_count,
                        }
                    )

            total_steps, total_how_far, total_rescue_count = 0, 0.0, 0

            for t in args.track:
                steps, how_far, rescue_count = pytux.rollout(
                    t,
                    parameterized_control,
                    max_frames=args.max_steps,
                    verbose=args.verbose,
                    train_callback=train_callback,
                    break_on_rescue=True,
                    planner=planner,
                )
                total_steps += steps
                total_how_far += how_far
                total_rescue_count += rescue_count

            PyTux._singleton = None
            pytux.close()
            del pytux
            return {"steps": total_steps, "how_far": total_how_far, "rescue_count": total_rescue_count}

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
        if args.model == "simple_2":
            best_points = [CURRENT_BEST_SIMPLE_2_PARAMS]
            param_space = {
                "aim_steering": tune.uniform(0.0, 3.0),
                "vel_steering": tune.uniform(0.0, 1.0),
                "drift_cutoff": tune.uniform(0.5, 1.0),
                "brake_cutoff": tune.uniform(0.2, 1.0),
                "vel_max": tune.uniform(10.0, 25.0),
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
    parser.add_argument("-p", "--planner", action="store_true")
    parser.add_argument("--search_alg", default="random")
    parser.add_argument("--model", default="simple")
    args = parser.parse_args()
    if args.test_run:
        test_controller(args)
    else:
        tune_controller(args)
