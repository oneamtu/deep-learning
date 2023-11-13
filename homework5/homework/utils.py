import numpy as np
import pystk

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

# import ray

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = "drive_data"
TRAIN_DATASET_PATH = f"{DATASET_PATH}/train"


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=TRAIN_DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path

        self.data = []
        count = 0
        for f in glob(path.join(dataset_path, "*.csv")):
            count += 1
            i = Image.open(f.replace(".csv", ".png"))
            i.load()
            # TODO: filter upstream
            coords = np.loadtxt(f, dtype=np.float32, delimiter=",")
            if coords[0] < -1 or coords[0] > 1 or coords[1] < -1 or coords[1] > 1:
                continue
            self.data.append((i, coords))
        print(f"{dataset_path} : {len(self.data)} valid points/{count} total")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=TRAIN_DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def split_data_files(dataset_path=DATASET_PATH, perc=0.2):
    import os
    import random
    import shutil

    train_dir = f"{dataset_path}/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        [shutil.move(os.path.join(dataset_path, file_path), train_dir) for file_path in os.listdir(dataset_path)]

    files = os.listdir(train_dir)
    files = [f for f in files if f.endswith(".png")]

    random.shuffle(files)
    slice_point = int(len(files) * perc)

    valid_dir = f"{dataset_path}/valid"
    if os.path.exists(valid_dir):
        return

    os.makedirs(valid_dir)

    for f in files[:slice_point]:
        shutil.move(os.path.join(train_dir, f), valid_dir)
        shutil.move(os.path.join(train_dir, f.replace(".png", ".csv")), valid_dir)


class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96, graphics=True):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        if graphics:
            self.config = pystk.GraphicsConfig.hd()
        else:
            self.config = pystk.GraphicsConfig.none()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1], distance % track.path_distance[-1, 1]) % len(
            track.path_nodes
        )
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([p[0] / p[-1], -p[1] / p[-1]])

    def rollout(
        self,
        track,
        controller,
        planner=None,
        max_frames=1000,
        verbose=False,
        data_callback=None,
        train=None,
        break_on_rescue=False,
        filename="test.mp4",
    ):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0
        rescue_count = 0
        dist = 0.
        total_accuracy = 0

        if verbose:
            import matplotlib.pyplot as plt
            import io

            fig, ax = plt.subplots(1, 1)
            frames = []

        for t in range(max_frames):
            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                # if verbose:
                print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            aim_point_world = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                gt_aim_point_image = aim_point_image
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
                dist = np.linalg.norm(aim_point_image - gt_aim_point_image)
                total_accuracy += dist < 5e-2

            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True
                rescue_count += 1
                if rescue_count > 0 and break_on_rescue:
                    break

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(
                    plt.Circle(WH2 * (1 + self._to_image(kart.location, proj, view)), 2, ec="b", fill=False, lw=1.5)
                )
                ax.add_artist(
                    plt.Circle(WH2 * (1 + self._to_image(aim_point_world, proj, view)), 2, ec="r", fill=False, lw=1.5)
                )
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(WH2 * (1 + aim_point_image), 2, ec="g", fill=False, lw=1.5))
                plt.pause(1e-3)

                text = "\n".join(
                    (
                        f"Im Points: X: {aim_point_image[0]:.4f}, Y: {aim_point_image[1]:.4f}, Vel: {current_vel:.4f}",
                        f"Last action: A: {action.acceleration:.4f}, B: {action.brake}, D: {action.drift}, S: {action.steer:.4f}",
                        f"steps: {t}, how_far: {(kart.overall_distance / track.length):.4f}, rescue_count: {rescue_count}",
                        f"accuracy: {dist}, total accuracy; {(total_accuracy / t):.2f}",
                    )
                )
                ax.text(0.5, 0.5, text, alpha=0.8, verticalalignment="top")

                with io.BytesIO() as buff:
                    fig.savefig(buff, format="raw")
                    buff.seek(0)
                    data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                im = data.reshape((int(h), int(w), -1))

                frames.append(im)

            self.k.step(action)
            t += 1
            how_far = kart.overall_distance / track.length

            if t < 100 and how_far > 0.8:
                how_far = 0

            if train is not None and t % 10 == 0:
                train.report(
                    {"steps": t, "how_far": how_far, "rescue_count": rescue_count}
                )
            if False:
                text = "\n".join(
                    (
                        f"Im Points: X: {aim_point_image[0]:.4f}, Y: {aim_point_image[1]:.4f}, Vel: {current_vel:.4f}",
                        f"Last action: A: {action.acceleration:.4f}, B: {action.brake}, D: {action.drift}, S: {action.steer:.4f}",
                        f"steps: {t}, how_far: {kart.overall_distance / track.length:.4f}, rescue_count: {rescue_count}",
                    )
                )
                print(text)

        if verbose:
            import imageio

            imageio.mimwrite(filename, frames, fps=30, bitrate=1000000)

        return t, how_far, rescue_count

    # rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5) for i in range(10)]
    # def rollout_many(many_agents, **kwargs):
    #     ray_data = []
    #     for i, agent in enumerate(many_agents):
    #         ray_data.append(rollouts[i % len(rollouts)].__call__.remote(agent, **kwargs) )
    #     return ray.get(ray_data)

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()


if __name__ == "__main__":
    from .controller import control
    from argparse import ArgumentParser
    from os import makedirs

    def noisy_control(aim_pt, vel):
        return control(aim_pt + np.random.randn(*aim_pt.shape) * aim_noise, vel + np.random.randn() * vel_noise)

    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument("track", nargs="+")
    parser.add_argument("-o", "--output", default=DATASET_PATH)
    parser.add_argument("-n", "--n_images", default=10000, type=int)
    parser.add_argument("-m", "--steps_per_track", default=20000, type=int)
    parser.add_argument("--aim_noise", default=0.1, type=float)
    parser.add_argument("--vel_noise", default=5, type=float)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass
    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = 0, 0

        def collect(_, im, pt):
            from PIL import Image
            from os import path

            global n
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(args.output, track + "_%05d" % id)
                Image.fromarray(im).save(fn + ".png")
                with open(fn + ".csv", "w") as f:
                    f.write("%0.1f,%0.1f" % tuple(pt))
            n += 1

        while n < args.steps_per_track:
            steps, how_far, rescues = pytux.rollout(
                track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect
            )
            print(track, steps, how_far, rescues)
            # Add noise after the first round
            aim_noise, vel_noise = args.aim_noise, args.vel_noise
    pytux.close()
