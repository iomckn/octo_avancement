import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class AlohaRldsDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary": tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                        ),
                        "proprio": tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                    ),
                    "reward": tf.float32,
                    "discount": tf.float32,
                    "is_first": tf.bool,
                    "is_last": tf.bool,
                    "is_terminal": tf.bool,
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        data_dir = r"C:\Users\nkeungueu\Desktop\atelier_ia\octo\aloha_dataset"

        return {
            "train": self._generate_examples(data_dir),
        }

    def _generate_examples(self, path):

        files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".hdf5")
        ])

        for episode_idx, file_path in enumerate(files):

            with h5py.File(file_path, "r") as f:

                actions = f["action"][:]
                qpos = f["observations"]["qpos"][:]
                images = f["observations"]["images"]["top"][:]

                steps = []

                for i in range(len(actions)):

                    step = {
                        "observation": {
                            "image_primary": images[i],
                            "proprio": qpos[i].astype(np.float32),
                        },
                        "action": actions[i].astype(np.float32),
                        "reward": np.float32(0.0),
                        "discount": np.float32(1.0),
                        "is_first": i == 0,
                        "is_last": i == len(actions) - 1,
                        "is_terminal": i == len(actions) - 1,
                        "language_instruction": (
                            "pick up the cube and hand it over"
                        ),
                    }

                    steps.append(step)

                yield episode_idx, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": file_path,
                    },
                }


if __name__ == "__main__":

    builder = AlohaRldsDataset()

    builder.download_and_prepare()

    print("Dataset RLDS généré avec succès.")