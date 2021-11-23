"""Module to generate motion from music."""

import os
import subprocess
import time
import pickle
import librosa
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from mint.core import model_builder
from mint.utils import config_util
from scipy.spatial.transform import Rotation as R

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', None, 'Directory to find training checkpoints and logs')
flags.DEFINE_string('config_path', None, 'Path to the model config file.')
flags.DEFINE_string('audio_path', None, 'Path to the music audio file.')
flags.DEFINE_string('motion_seed', None, 'Path to the motion seed file.')
flags.DEFINE_string('smpl_path', None, 'Path to the Male SMPL model file')


def _log(message: str):
    """Logs `message` to the `info` log, and also prints to stdout."""
    logging.info(message)
    # print(message)


def _get_audio_feature(audio_path):
    """Extract feature from audio file."""
    assert os.path.exists(audio_path), f'File {audio_path} does not exist!'
    _log(f"process music feature {audio_path}.")

    audio_feature = None
    if audio_path.endswith(".npy"):
        # load audio feature from feature data file
        with open(audio_path, 'rb') as f:
            audio_feature = np.load(f)
            audio_feature = np.array(audio_feature)  # (N, 35)
            f.close()
    else:
        # fetch audio feature from wav audio file
        FPS = 60
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH

        data, _ = librosa.load(FLAGS.audio_path, sr=SR)
        # (seq_len,)
        envelope = librosa.onset.onset_strength(data, sr=SR)
        # (seq_len, 20)
        mfcc = librosa.feature.mfcc(data, n_mfcc=20, sr=SR).T
        # (seq_len, 12)
        chroma = librosa.feature.chroma_cens(
            data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
        ).T
        # (seq_len,)
        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
        )
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0
        # (seq_len,)
        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH, tightness=100.0
        )
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0
        # concat feature (?, 35)
        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma,
            peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)

    # reshape to (1, ?, 35)
    audio_feature = audio_feature[np.newaxis, :, :]
    _log(audio_feature.shape)

    return audio_feature


def _get_motion_feature(motion_seed):
    """Get motion feature."""
    assert os.path.exists(motion_seed), f'File {motion_seed} does not exist!'
    _log(f"process seed motion feature {motion_seed}.")

    with open(motion_seed, 'rb') as f:
        data = pickle.load(f)
        f.close()

    smpl_poses = data['smpl_poses']         # (N, 24, 3)
    smpl_scaling = data['smpl_scaling']     # (1,)
    smpl_trans = data['smpl_trans']         # (N, 3)

    # fetch (120, 225) motion data
    smpl_trans /= smpl_scaling
    smpl_poses = R.from_rotvec(smpl_poses.reshape(-1, 3)).as_matrix().reshape(smpl_poses.shape[0], -1)
    smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)
    smpl_motion = tf.pad(smpl_motion[0:120, :], [[0, 0], [6, 0]])
    smpl_motion.set_shape([120, 225])

    # reshape to (1, 120, 225)
    smpl_motion = smpl_motion[np.newaxis, :, :].numpy()
    _log(smpl_motion.shape)
    return smpl_motion


def main(_):
    """Generate motion."""
    start = time.perf_counter()
    configs = config_util.get_configs_from_pipeline_file(FLAGS.config_path)
    model_config = configs['model']
    # restore model
    model_ = model_builder.build(model_config, False)
    _log(f"restore model from {FLAGS.config_path}.")
    # model_.build({
    #     "audio_input": (None, None, 35),
    #     "motion_input": (None, 120, 225)
    # })
    # _log(model_.summary())
    # restore weight
    checkpoint = tf.train.Checkpoint(model=model_)
    latest = tf.train.latest_checkpoint(FLAGS.model_dir)
    checkpoint.restore(latest).expect_partial()
    _log(f"restored weight from {latest}.")
    # generate motion
    features = {
        "motion_input": _get_motion_feature(FLAGS.motion_seed),
        "audio_input": _get_audio_feature(FLAGS.audio_path)
    }
    _log(f"generate motion from music.")
    output = model_.infer_auto_regressive(features, steps=5000)
    # [batch_size, motion_seq_length + steps, motion_feature_dimension]
    output = tf.concat([features["motion_input"], output], axis=1)
    _log(f"motion result shape {output.shape}")
    output = output[0].numpy()
    save_path = FLAGS.audio_path[:-4]+'-motion.npy'
    _log(f"save motion result to {save_path}.")
    # [steps, motion_feature_dimension]
    np.save(save_path, output)
    _log(f"motion generate finished, cost {time.perf_counter()-start} sec.")

    # export fbx file
    # subprocess.call([
    #     "python", os.path.join(os.getcwd(), "./exporter.py"),
    #     f"--smpl_path={os.path.join(os.getcwd(), FLAGS.smpl_path)}",
    #     f"--motion_path={os.path.join(os.getcwd(), save_path)}"], shell=True
    # )


if __name__ == '__main__':
    app.run(main)
