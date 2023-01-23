# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a humanoid to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env


class Robot(env.Env):

  def __init__(self,
               forward_reward_weight=1.25,
               ctrl_cost_weight=0.1,
               healthy_reward=5.0,
               terminate_when_unhealthy=True,
               healthy_z_range=(0.8, 2.1),
               reset_noise_scale=1e-2,
               exclude_current_positions_from_observation=True,
               legacy_spring=False,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING 
    super().__init__(config=config, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)

    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp, self.sys.info(qp), jp.zeros(self.action_size))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)

    com_before = self._center_of_mass(state.qp)
    com_after = self._center_of_mass(qp)
    velocity = (com_after - com_before) / self.sys.config.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(qp, info, action)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info,
               action: jp.ndarray) -> jp.ndarray:
    """Observe humanoid body position, velocities, and angles."""
    angle_vels = [j.angle_vel(qp) for j in self.sys.joints]

    # qpos: position and orientation of the torso and the joint angles.
    joint_angles = [angle for angle, _ in angle_vels]
    if self._exclude_current_positions_from_observation:
      qpos = [qp.pos[0, 2:], qp.rot[0]] + joint_angles
    else:
      qpos = [qp.pos[0], qp.rot[0]] + joint_angles

    # qvel: velocity of the torso and the joint angle velocities.
    joint_velocities = [vel for _, vel in angle_vels]
    qvel = [qp.vel[0], qp.ang[0]] + joint_velocities

    # center of mass obs:
    com = self._center_of_mass(qp)
    mass_sum = jp.sum(self.sys.body.mass[:-1])

    def com_vals(body, qp):
      d = qp.pos - com
      com_inr = body.mass * jp.eye(3) * jp.norm(d) ** 2
      com_inr += jp.diag(body.inertia) - jp.outer(d, d)
      com_vel = body.mass * qp.vel / mass_sum
      com_ang = jp.cross(d, qp.vel) / (1e-7 + jp.norm(d) ** 2)

      return com_inr, com_vel, com_ang

    com_inr, com_vel, com_ang = jp.vmap(com_vals)(self.sys.body, qp)
    cinert = [com_inr[:-1].ravel()]
    cvel = [com_vel[:-1].ravel(), com_ang[:-1].ravel()]

    # actuator forces
    qfrc_actuator = []
    for act in self.sys.actuators:
      torque = jp.take(action, act.act_index)
      torque = torque.reshape(torque.shape[:-2] + (-1,))
      torque *= jp.repeat(act.strength, act.act_index.shape[-1])
      qfrc_actuator.append(torque)

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # can be calculated in brax like so:
    # cfrc = [
    #     jp.clip(info.contact.vel, -1, 1),
    #     jp.clip(info.contact.ang, -1, 1)
    # ]
    # flatten bottom dimension
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
    # then add it to the jp.concatenate below

    return jp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator)

  def _center_of_mass(self, qp):
    mass, pos = self.sys.body.mass[:-1], qp.pos[:-1]
    return jp.sum(jp.vmap(jp.multiply)(mass, pos), axis=0) / jp.sum(mass)

  def _noise(self, rng):
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)

_SYSTEM_CONFIG_SPRING = """
bodies {
  name: "pelvis"
  colliders {
    position {
      z: 0.07000000029802322
    }
    sphere {
      radius: 0.09000000357627869
    }
  }
  colliders {
    position {
      z: 0.20499999821186066
    }
    sphere {
      radius: 0.07000000029802322
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 9.995593070983887
}
bodies {
  name: "torso"
  colliders {
    position {
      z: 0.11999999731779099
    }
    sphere {
      radius: 0.10999999940395355
    }
  }
  colliders {
    position {
      x: -0.011423749849200249
      y: -0.08697725087404251
      z: 0.2332068532705307
    }
    rotation {
      x: 83.88849639892578
      y: -7.44045352935791
      z: 6.68812370300293
    }
    capsule {
      radius: 0.04500000178813934
      length: 0.17357417941093445
    }
  }
  colliders {
    position {
      x: -0.011423749849200249
      y: 0.08697725087404251
      z: 0.2332068532705307
    }
    rotation {
      x: -83.88849639892578
      y: -7.44045352935791
      z: -6.68812370300293
    }
    capsule {
      radius: 0.04500000178813934
      length: 0.17357417941093445
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 12.01148796081543
}
bodies {
  name: "head"
  colliders {
    position {
      z: 0.17499999701976776
    }
    sphere {
      radius: 0.0949999988079071
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.8822643756866455
}
bodies {
  name: "right_upper_arm"
  colliders {
    position {
      z: -0.14000000059604645
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.04500000178813934
      length: 0.27000001072883606
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.4993313550949097
}
bodies {
  name: "right_lower_arm"
  colliders {
    position {
      z: -0.11999999731779099
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.03999999910593033
      length: 0.2150000035762787
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.9996799230575562
}
bodies {
  name: "right_hand"
  colliders {
    position {
    }
    sphere {
      radius: 0.03999999910593033
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.49997401237487793
}
bodies {
  name: "left_upper_arm"
  colliders {
    position {
      z: -0.14000000059604645
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.04500000178813934
      length: 0.27000001072883606
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.4993313550949097
}
bodies {
  name: "left_lower_arm"
  colliders {
    position {
      z: -0.11999999731779099
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.03999999910593033
      length: 0.2150000035762787
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.9996799230575562
}
bodies {
  name: "left_hand"
  colliders {
    position {
    }
    sphere {
      radius: 0.03999999910593033
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.49997401237487793
}
bodies {
  name: "right_thigh"
  colliders {
    position {
      z: -0.20999999344348907
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.054999999701976776
      length: 0.4099999964237213
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.502291679382324
}
bodies {
  name: "right_shin"
  colliders {
    position {
      z: -0.20000000298023224
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.4099999964237213
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.9997496604919434
}
bodies {
  name: "right_foot"
  colliders {
    position {
      x: 0.04500000178813934
      z: -0.02250000089406967
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.0885000005364418
        y: 0.04500000178813934
        z: 0.027499999850988388
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.999687135219574
}
bodies {
  name: "left_thigh"
  colliders {
    position {
      z: -0.20999999344348907
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.054999999701976776
      length: 0.4099999964237213
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.502291679382324
}
bodies {
  name: "left_shin"
  colliders {
    position {
      z: -0.20000000298023224
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.4099999964237213
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.9997496604919434
}
bodies {
  name: "left_foot"
  colliders {
    position {
      x: 0.04500000178813934
      z: -0.02250000089406967
    }
    rotation {
      x: -0.0
      z: -0.0
    }
    box {
      halfsize {
        x: 0.0885000005364418
        y: 0.04500000178813934
        z: 0.027499999850988388
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.999687135219574
}
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "abdomen_x"
  parent: "pelvis"
  child: "torso"
  parent_offset {
    z: 0.23615099489688873
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -60.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "abdomen_y"
  parent: "pelvis"
  child: "torso"
  parent_offset {
    z: 0.23615099489688873
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -60.0
    max: 90.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "abdomen_z"
  parent: "pelvis"
  child: "torso"
  parent_offset {
    z: 0.23615099489688873
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -50.0
    max: 50.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "neck_x"
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.22389400005340576
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -50.0
    max: 50.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "neck_y"
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.22389400005340576
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -40.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "neck_z"
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.22389400005340576
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -45.0
    max: 45.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_shoulder_x"
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.024049999192357063
    y: -0.18310999870300293
    z: 0.2434999942779541
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -180.0
    max: 45.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_shoulder_y"
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.024049999192357063
    y: -0.18310999870300293
    z: 0.2434999942779541
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -180.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_shoulder_z"
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.024049999192357063
    y: -0.18310999870300293
    z: 0.2434999942779541
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -90.0
    max: 90.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_elbow"
  parent: "right_upper_arm"
  child: "right_lower_arm"
  parent_offset {
    z: -0.2747879922389984
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -160.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "$right_lower_arm.right_hand"
  parent: "right_lower_arm"
  child: "right_hand"
  parent_offset {
    z: -0.2589470148086548
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -5.0
    max: 5.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_shoulder_x"
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.024049999192357063
    y: 0.18310999870300293
    z: 0.2434999942779541
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -45.0
    max: 180.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_shoulder_y"
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.024049999192357063
    y: 0.18310999870300293
    z: 0.2434999942779541
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -180.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_shoulder_z"
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.024049999192357063
    y: 0.18310999870300293
    z: 0.2434999942779541
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -90.0
    max: 90.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_elbow"
  parent: "left_upper_arm"
  child: "left_lower_arm"
  parent_offset {
    z: -0.2747879922389984
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -160.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "$left_lower_arm.left_hand"
  parent: "left_lower_arm"
  child: "left_hand"
  parent_offset {
    z: -0.2589470148086548
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -5.0
    max: 5.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_hip_x"
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.08488699793815613
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -60.0
    max: 15.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_hip_y"
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.08488699793815613
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -140.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_hip_z"
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.08488699793815613
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -60.0
    max: 35.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_knee"
  parent: "right_thigh"
  child: "right_shin"
  parent_offset {
    z: -0.42154601216316223
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    max: 160.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_ankle_x"
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.4098699986934662
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -30.0
    max: 30.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_ankle_y"
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.4098699986934662
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -55.0
    max: 55.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_ankle_z"
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.4098699986934662
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_hip_x"
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.08488699793815613
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -15.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_hip_y"
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.08488699793815613
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -140.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_hip_z"
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.08488699793815613
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -35.0
    max: 60.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_knee"
  parent: "left_thigh"
  child: "left_shin"
  parent_offset {
    z: -0.42154601216316223
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    max: 160.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_ankle_x"
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.4098699986934662
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angular_damping: 30
  angle_limit {
    min: -30.0
    max: 30.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_ankle_y"
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.4098699986934662
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angular_damping: 30
  angle_limit {
    min: -55.0
    max: 55.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_ankle_z"
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.4098699986934662
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angular_damping: 30
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
actuators {
  name: "abdomen_x"
  joint: "abdomen_x"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "abdomen_y"
  joint: "abdomen_y"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "abdomen_z"
  joint: "abdomen_z"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "neck_x"
  joint: "neck_x"
  strength: 20.0
  torque {
  }
}
actuators {
  name: "neck_y"
  joint: "neck_y"
  strength: 20.0
  torque {
  }
}
actuators {
  name: "neck_z"
  joint: "neck_z"
  strength: 20.0
  torque {
  }
}
actuators {
  name: "right_shoulder_x"
  joint: "right_shoulder_x"
  strength: 70.0
  torque {
  }
}
actuators {
  name: "right_shoulder_y"
  joint: "right_shoulder_y"
  strength: 70.0
  torque {
  }
}
actuators {
  name: "right_shoulder_z"
  joint: "right_shoulder_z"
  strength: 70.0
  torque {
  }
}
actuators {
  name: "right_elbow"
  joint: "right_elbow"
  strength: 60.0
  torque {
  }
}
actuators{
  name: "$right_lower_arm.right_hand"
  joint: "$right_lower_arm.right_hand"
  strength: 300.0
  torque{
  }
}
actuators {
  name: "left_shoulder_x"
  joint: "left_shoulder_x"
  strength: 70.0
  torque {
  }
}
actuators {
  name: "left_shoulder_y"
  joint: "left_shoulder_y"
  strength: 70.0
  torque {
  }
}
actuators {
  name: "left_shoulder_z"
  joint: "left_shoulder_z"
  strength: 70.0
  torque {
  }
}
actuators {
  name: "left_elbow"
  joint: "left_elbow"
  strength: 60.0
  torque {
  }
}
actuators{
  name: "$left_lower_arm.left_hand"
  joint: "$left_lower_arm.left_hand"
  strength: 300.0
  torque{
  }
}
actuators {
  name: "right_hip_x"
  joint: "right_hip_x"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "right_hip_z"
  joint: "right_hip_z"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "right_hip_y"
  joint: "right_hip_y"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "right_knee"
  joint: "right_knee"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "right_ankle_x"
  joint: "right_ankle_x"
  strength: 50.0
  torque {
  }
}
actuators {
  name: "right_ankle_y"
  joint: "right_ankle_y"
  strength: 50.0
  torque {
  }
}
actuators {
  name: "right_ankle_z"
  joint: "right_ankle_z"
  strength: 50.0
  torque {
  }
}
actuators {
  name: "left_hip_x"
  joint: "left_hip_x"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "left_hip_z"
  joint: "left_hip_z"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "left_hip_y"
  joint: "left_hip_y"
  strength: 125.0
  torque {
  }
}
actuators {
  name: "left_knee"
  joint: "left_knee"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "left_ankle_x"
  joint: "left_ankle_x"
  strength: 50.0
  torque {
  }
}
actuators {
  name: "left_ankle_y"
  joint: "left_ankle_y"
  strength: 50.0
  torque {
  }
}
actuators {
  name: "left_ankle_z"
  joint: "left_ankle_z"
  strength: 50.0
  torque {
  }
}
collide_include {
    first: "floor"
    second: "left_shin"
}
collide_include {
    first: "floor"
    second: "right_shin"
}
collide_include {
    first: "floor"
    second: "left_foot"
}
collide_include {
    first: "floor"
    second: "right_foot"
}
defaults{
  angles{
    name: "left_knee"
    angle {
      x:40
    }
  }
  angles{
    name: "right_knee"
    angle {
      x:40
    }
  }
  angles{
    name: "right_hip_z"
    angle {
    }
  }
  angles{
    name: "left_hip_z"
    angle {
    }
  }
}
friction: 1.0
gravity { z: -9.8 }
angular_damping: -0.05
dt: 0.05
substeps: 30
dynamics_mode: "pbd"
"""