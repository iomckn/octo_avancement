import sys
from functools import partial

import gym
import jax
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import cv2
# Chemin vers ton dossier qui contient aloha_sim_env.py
sys.path.append(r"C:\Users\nkeungueu\Desktop\atelier_ia")

# Important : cet import enregistre "aloha-sim-cube-v0"
import aloha_sim_env  # noqa

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng


MODEL_PATH = r"C:\Users\nkeungueu\Desktop\atelier_ia\octo\checkpoints\aloha_finetuned"

print("Chargement du modèle...")
model = OctoModel.load_pretrained(MODEL_PATH)

print("Création environnement...")
env = gym.make("aloha-sim-cube-v0")

env = NormalizeProprio(env, model.dataset_statistics)
env = HistoryWrapper(env, horizon=1)
env = RHCWrapper(env, exec_horizon=50)

policy_fn = supply_rng(
    partial(
        model.sample_actions,
        unnormalization_statistics=model.dataset_statistics["action"],
    )
)

obs, info = env.reset()
task = model.create_tasks(texts=["pick up the cube and hand it over"])

plt.ion()
fig, ax = plt.subplots()

video_path = "simulation_octo.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(
    video_path,
    fourcc,
    20.0,
    (640, 480)
)

for step in range(400):
    print("Step", step, "avant prédiction")

    actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
    actions = np.array(actions[0])

    print("Step", step, "action prédite", actions.shape)

    obs, reward, done, trunc, info = env.step(actions)

    print("Step", step, "après step")
    print("action min/max:", actions.min(), actions.max())
    print("première action:", actions[0])

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    img = base_env._env.physics.render(
        height=480,
        width=640,
        camera_id="top"
    )
    frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    video.write(frame_bgr)

    print("Step", step, "image rendue")

    ax.clear()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Step {step} | reward={reward}")
    plt.pause(0.1)

video.release()

print("Vidéo sauvegardée :", video_path)
plt.ioff()
plt.show()