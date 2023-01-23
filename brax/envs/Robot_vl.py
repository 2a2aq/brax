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

"""Trains an ant to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env
from jax import lax


class Robot(env.Env):
  def __init__(self,
               forward_reward_weight=1.25,
               ctrl_cost_weight=0.1,
               healthy_reward=5.0,
               terminate_when_unhealthy=True,
               healthy_z_range=(0.6, 1.2),
               reset_noise_scale=1e-2,
               exclude_current_positions_from_observation=True,
               legacy_spring=True,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
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
    self._rng = rng

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
        forward_reward=state.qp.pos[0,0],
        reward_linvel=state.qp.pos[0,2],
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=qp.pos[0,0],
        y_position=qp.pos[0,2],
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

# TODO: name bodies in config according to mujoco xml
# pelvis mass 4/3pi*(0.09)^3*2226
# upper_waist mass 4/3pi*(0.07)^3*2226
# torso mass 4/3pi*(0.11)^3*1794
# right_clavicle length sqrt[(length fromto_x-x)^2+..]
# right_clavicle mass pir^2*l*density
# right_foot mass 8xyz*density
_SYSTEM_CONFIG = """
  bodies {
    name: "pelvis"
    colliders {
      capsule{
        radius: 0.09
        length: 0.18
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 6.7973760600
  }
  bodies {
    name: "upper_waist"
    colliders {
      capsule{
        radius: 0.07
        length: 0.14
      }
      inertia { x: 1.0 y: 1.0 z: 1.0 }
      mass : 3.1982167196
    }
  }
  bodies {
    name: "torso"
    colliders {
      capsule{
        radius: 0.11
        length: 0.22
      }
      inertia { x: 1.0 y: 1.0 z: 1.0 }
      mass : 10.0020518941
    }
  }
  bodies {
    name: "right_clavicle"
    colliders{
      rotation { z: -153.09 y: 87.23 }
      capsule{
        radius: 0.045
        length: 0.1822662381
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 1.2754804765
  }
  bodies {
    name: "left_clavicle"
    colliders{
      rotation { z: 153.09 y: 87.23 }
      capsule{
        radius: 0.045
        length: 0.1822662381
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 1.2754804765
  }
  bodies {
    name: "head"
    colliders{
      capsule{
        radius: 0.095
        length: 0.190
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 3.8822644860
  }
  bodies {
    name: "right_upper_arm"
    colliders{
      capsule{
        radius: 0.045
        length: 0.1800000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 1.1244985328
  }
  bodies {
    name: "right_lower_arm"
    colliders{
      capsule{
        radius: 0.04
        length: 0.1350000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 0.7165847179
  }
  bodies {
    name: "right_hand"
    colliders{
      capsule{
        radius: 0.04
        length: 0.08
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 0.4999739988
  }
  bodies {
    name: "left_upper_arm"
    colliders{
      capsule{
        radius: 0.045
        length: 0.1800000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 1.1244985328
  }
  bodies {
    name: "left_lower_arm"
    colliders{
      capsule{
        radius: 0.04
        length: 0.1350000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 0.7165847179
  }
  bodies {
    name: "left_hand"
    colliders{
      capsule{
        radius: 0.04
        length: 0.08
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 0.4999739988
  }
  bodies {
    name: "right_thigh"
    colliders{
      capsule{
        radius: 0.055
        length: 0.3000000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 3.6179130777
  }
  bodies {
    name: "right_shin"
    colliders{
      capsule{
        radius: 0.05
        length: 0.3100000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 2.4688205868
  }
  bodies {
    name: "right_foot"
    colliders{
      box{
        halfsize { x: 0.0885 y: 0.045 z: 0.0275}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 0.9996871500
  }
  bodies {
    name: "left_thigh"
    colliders{
      capsule{
        radius: 0.055
        length: 0.3000000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 3.6179130777
  }
  bodies {
    name: "left_shin"
    colliders{
      capsule{
        radius: 0.05
        length: 0.3100000000
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 2.4688205868
  }
  bodies {
    name: "left_foot"
    colliders{
      box{
        halfsize { x: 0.0885 y: 0.045 z: 0.0275}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass : 0.9996871500
  }
  bodies {
    name: "Ground"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  joints {
    name: "line25"
    parent_offset {z: 0.93}
    child_offset {z: 0.795}
    parent: "pelvis"
    child: "upper_waist"
    angular_damping: 60
  }
  joints {
    name: "line38"
    parent_offset {z: 0.126151}
    child_offset {z: 0.116151}
    parent: "torso
  }



  
  joints {
    name: "hip_1"
    parent_offset { x: 0.2 y: 0.2 }
    child_offset { x: -0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 1"
    angle_limit { min: -30.0 max: 30.0 }
    rotation { y: -90 }
    angular_damping: 20
  }
  joints {
    name: "ankle_1"
    parent_offset { x: 0.1 y: 0.1 }
    child_offset { x: -0.2 y: -0.2 }
    parent: "Aux 1"
    child: "$ Body 4"
    rotation: { z: 135 }
    angle_limit {
      min: 30.0
      max: 70.0
    }
    angular_damping: 20
  }
  joints {
    name: "hip_2"
    parent_offset { x: -0.2 y: 0.2 }
    child_offset { x: 0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 2"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_2"
    parent_offset { x: -0.1 y: 0.1 }
    child_offset { x: 0.2 y: -0.2 }
    parent: "Aux 2"
    child: "$ Body 7"
    rotation { z: 45 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "hip_3"
    parent_offset { x: -0.2 y: -0.2 }
    child_offset { x: 0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 3"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_3"
    parent_offset { x: -0.1 y: -0.1 }
    child_offset {
      x: 0.2
      y: 0.2
    }
    parent: "Aux 3"
    child: "$ Body 10"
    rotation { z: 135 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "hip_4"
    parent_offset { x: 0.2 y: -0.2 }
    child_offset { x: -0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 4"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_4"
    parent_offset { x: 0.1 y: -0.1 }
    child_offset { x: -0.2 y: 0.2 }
    parent: "Aux 4"
    child: "$ Body 13"
    rotation { z: 45 }
    angle_limit { min: 30.0 max: 70.0 }
    angular_damping: 20
  }
  actuators {
    name: "hip_1"
    joint: "hip_1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_1"
    joint: "ankle_1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_2"
    joint: "hip_2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_2"
    joint: "ankle_2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_3"
    joint: "hip_3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_3"
    joint: "ankle_3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_4"
    joint: "hip_4"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_4"
    joint: "ankle_4"
    strength: 350.0
    torque {}
  }
  friction: 1.0
  gravity { z: -9.8 }
  angular_damping: -0.05
  collide_include {
    first: "$ Torso"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 4"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 7"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 10"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 13"
    second: "Ground"
  }
  dt: 0.05
  substeps: 10
  dynamics_mode: "pbd"
  """

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
  name: "right_hip_y"
  joint: "right_hip_y"
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
  name: "left_hip_x"
  joint: "left_hip_x"
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
  name: "left_hip_z"
  joint: "left_hip_z"
  strength: 125.0
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
substeps: 10
dynamics_mode: "pbd"
"""
