"""Export motion frames to fbx file."""
"""ref to https://github.com/Arthur151/ROMP/blob/master/romp/exports/convert_fbx.py"""

import os
import pickle
import bpy
import time
import numpy as np
from absl import app
from absl import flags
from absl import logging
from math import radians
from scipy.spatial.transform import Rotation as R
from mathutils import Matrix, Vector, Quaternion

FLAGS = flags.FLAGS
flags.DEFINE_string('smpl_path', None, 'Path to the Male SMPL model file')
flags.DEFINE_string('motion_path', None, 'Path to the motion frames file.')

bone_name_from_index = {
    0: 'Pelvis',
    1: 'L_Hip',
    2: 'R_Hip',
    3: 'Spine1',
    4: 'L_Knee',
    5: 'R_Knee',
    6: 'Spine2',
    7: 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}


def _log(message: str):
    """Logs `message` to the `info` log, and also prints to stdout."""
    logging.info(message)
    # print(message)


def _load_data(motion_path):
    """Load motion data"""
    _log(f"load motion data from {motion_path}.")
    poses = None
    trans = None
    output = ""
    # load motion data from npy file
    if motion_path.endswith(".npy"):
        with open(motion_path, 'rb') as f:
            data = np.load(f)
            data = np.array(data)  # (N, 225)
            f.close()
        trans = data[:, 6:9]
        poses = data[:, 9:]
        poses = R.from_matrix(poses.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 72)
        output = motion_path.replace(".npy", "-mdl.fbx")
    # load motion data from pkl file
    elif motion_path.endswith(".pkl"):
        with open(motion_path, 'rb') as f:
            data = pickle.load(f)
            f.close()
        poses = data['smpl_poses']  # (N, 72)
        scaling = data['smpl_scaling']  # (1,)
        trans = data['smpl_trans'] / scaling  # (N, 3)
        output = motion_path.replace(".pkl", "-cap.fbx")
    _log(f"poses shape {poses.shape}")
    return poses, trans, output


def _setup_scene(smpl_path, fps_target=60):
    """Setup Scene"""
    scene = bpy.data.scenes['Scene']

    ###########################
    # Engine independent setup
    ###########################

    scene.render.fps = fps_target

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # Import gender specific .fbx template file
    bpy.ops.import_scene.fbx(filepath=smpl_path)


def _rodrigues(rotvec):
    """Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues"""
    """Source: smpl/plugins/blender/corrective_bpy_sh.py"""
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=np.float32)
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


def _process_pose(current_frame, pose, trans, pelvis_position):
    """Process single pose into keyframed bone orientations"""
    rod_rots = pose.reshape(24, 3)

    mat_rots = [_rodrigues(rod_rot) for rod_rot in rod_rots]

    # Set the location of the Pelvis bone to the translation parameter
    armature = bpy.data.objects['Armature']
    bones = armature.pose.bones

    # Pelvis: X-Right, Y-Up, Z-Forward (Blender -Y)

    # Set absolute pelvis location relative to Pelvis bone head
    bones[bone_name_from_index[0]].location = Vector((100 * trans[1], 100 * trans[2], 100 * trans[0])) \
                                              - pelvis_position
    # bones['Root'].location = Vector(trans)
    bones[bone_name_from_index[0]].keyframe_insert('location', frame=current_frame)

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue

        bone = bones[bone_name_from_index[index]]
        bone_rotation = Matrix(mat_rot).to_quaternion()
        quat_x_90_cw = Quaternion(Vector([1.0, 0.0, 0.0]), radians(-90))
        quat_x_n135_cw = Quaternion(Vector([1.0, 0.0, 0.0]), radians(-135))
        quat_x_p45_cw = Quaternion(Vector([1.0, 0.0, 0.0]), radians(45))
        quat_y_90_cw = Quaternion(Vector([0.0, 1.0, 0.0]), radians(-90))
        quat_z_90_cw = Quaternion(Vector([0.0, 0.0, 1.0]), radians(-90))

        if index == 0:
            # Rotate pelvis so that avatar stands upright and looks along negative Y avis
            bone.rotation_quaternion = (quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
        else:
            bone.rotation_quaternion = bone_rotation

        bone.keyframe_insert('rotation_quaternion', frame=current_frame)

    return


def _process_poses(scene, poses, trans, pelvis_position):
    """Process all the poses from the pose file"""
    index = 0
    frame = 1
    offset = np.array([0.0, 0.0, 0.0], dtype=float)

    while index < poses.shape[0]:
        # Go to new frame
        scene.frame_set(frame)
        _process_pose(frame, poses[index], (trans[index] - offset), pelvis_position)
        index += 1
        frame += 1

    return frame


def _rotate_armature():
    """Rotate the root bone on the Y axis by -90 on export. Otherwise it may be rotated incorrectly"""
    # Switch to Pose Mode
    bpy.ops.object.posemode_toggle()

    # Find the Armature & Bones
    ob = bpy.data.objects['Armature']
    armature = ob.data
    bones = armature.bones
    rootbone = bones[0]

    # Find the Root bone
    for bone in bones:
        if "avg_root" in bone.name:
            rootbone = bone

    rootbone.select = True

    # Rotate the Root bone by 90 euler degrees on the Y axis. Set --rotate_Y=False if the rotation is not needed.
    bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False, use_proportional_projected=False,
                             release_confirm=True)
    # Revert back to Object Mode
    bpy.ops.object.posemode_toggle()


def _export_animated_mesh(output):
    """Exporting to FBX binary (.fbx)"""
    # Fix Rotation
    # _rotate_armature()
    # Select only skinned mesh and rig
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Armature'].select_set(True)
    bpy.data.objects['Armature'].children[0].select_set(True)
    bpy.ops.export_scene.fbx(filepath=output, use_selection=True, add_leaf_bones=False)


def export(motion_path, smpl_path):
    """Export motion to fbx file"""
    start = time.perf_counter()

    assert os.path.exists(motion_path), f"File {motion_path} does not exist!"
    # load motion data
    poses, trans, output = _load_data(motion_path)

    # Male SMPL model path
    for k, v in bone_name_from_index.items():
        bone_name_from_index[k] = 'm_avg_' + v

    _log("import smpl model and setup scene")
    # Setup Scene
    _setup_scene(smpl_path)
    scene = bpy.data.scenes['Scene']
    scene.frame_end = poses.shape[0]

    # Retrieve pelvis world position.
    # Unit is [cm] due to Armature scaling.
    # Need to make copy since reference will change when bone location is modified.
    # armaturee = bpy.data.armatures[0]
    ob = bpy.data.objects['Armature']
    armature = ob.data

    bpy.ops.object.mode_set(mode='EDIT')
    # get specific bone name 'Bone'
    pelvis_bone = armature.edit_bones[bone_name_from_index[0]]
    # pelvis_bone = armature.edit_bones['f_avg_Pelvis']
    pelvis_position = Vector(pelvis_bone.head)
    bpy.ops.object.mode_set(mode='OBJECT')

    _log("process poses")
    frames = _process_poses(scene, poses, trans, pelvis_position)

    _log(f"export fbx file to {output}")
    _export_animated_mesh(output)
    _log(f"animation export finished, process {frames} frames, cost {time.perf_counter() - start} sec.")


def main(_):
    export(FLAGS.motion_path, FLAGS.smpl_path)


if __name__ == '__main__':
    app.run(main)
