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


class Robot(env.Env):



  """
  ### Description

  This environment is based on the environment introduced by Schulman, Moritz,
  Levine, Jordan and Abbeel in
  ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).

  The ant is a 3D robot consisting of one torso (free rotational body) with four
  legs attached to it with each leg having two links.

  The goal is to coordinate the four legs to move in the forward (right)
  direction by applying torques on the eight hinges connecting the two links of
  each leg and the torso (nine parts and eight hinges).

  ### Action Space

  The agent take a 8-element vector for actions.

  The action space is a continuous `(action, action, action, action, action,
  action, action, action)` all in `[-1, 1]`, where `action` represents the
  numerical torques applied at the hinge joints.

  | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
  |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
  | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
  | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | ankle_1 (front_left_leg)         | hinge | torque (N m) |
  | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
  | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | ankle_2 (front_right_leg)        | hinge | torque (N m) |
  | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
  | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | ankle_3 (back_leg)               | hinge | torque (N m) |
  | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
  | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | ankle_4 (right_back_leg)         | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  ant, followed by the velocities of those individual parts (their derivatives)
  with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(27,)` where the elements correspond to the following:

  | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
  |-----|--------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
  | 1   | w-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 2   | x-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 3   | y-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 4   | z-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 5   | angle between torso and first link on front left             | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 6   | angle between the two links on the front left                | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 7   | angle between torso and first link on front right            | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 8   | angle between the two links on the front right               | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 9   | angle between torso and first link on back left              | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 10  | angle between the two links on the back left                 | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 11  | angle between torso and first link on back right             | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 12  | angle between the two links on the back right                | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
  | 13  | x-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 14  | y-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 15  | z-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 16  | x-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 17  | y-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 18  | z-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 19  | angular velocity of angle between torso and front left link  | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 20  | angular velocity of the angle between front left links       | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 21  | angular velocity of angle between torso and front right link | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 22  | angular velocity of the angle between front right links      | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 23  | angular velocity of angle between torso and back left link   | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 24  | angular velocity of the angle between back left links        | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 25  | angular velocity of angle between torso and back right link  | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 26  | angular velocity of the angle between back right links       | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |

  The (x,y,z) coordinates are translational DOFs while the orientations are
  rotational DOFs expressed as quaternions.

  If use_contact_forces=True, contact forces are added to the observation:
  external forces and torques applied to the center of mass of each of the
  links. 60 extra dimensions: the torso link, 8 leg links, and the ground link
  (10 total), with 6 external forces each (force x, y, z, torque x, y, z).

  ### Rewards

  The reward consists of three parts:

  - *reward_survive*: Every timestep that the ant is alive, it gets a reward of
    1.
  - *reward_forward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.05*. This reward would be
    positive if the ant moves forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the ant if it takes actions
    that are too large. It is measured as *coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.5.
  - *contact_cost*: A negative reward for penalising the ant if the external
    contact force is too large. It is calculated *0.5 * 0.001 *
    sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

  ### Starting State

  All observations start in state (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a
  uniform noise in the range of [-0.1, 0.1] added to the positional values and
  standard normal noise with 0 mean and 0.1 standard deviation added to the
  velocity values for stochasticity.

  Note that the initial z coordinate is intentionally selected to be slightly
  high, thereby indicating a standing up ant. The initial orientation is
  designed to make it face forward as well.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The y-orientation (index 2) in the state is **not** in the range
     `[0.2, 1.0]`

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder).

  ```
  env = gym.make('Ant-v2')
  ```

  v3, v4, and v5 take gym.make kwargs such as ctrl_cost_weight,
  reset_noise_scale etc.

  ```
  env = gym.make('Ant-v5', ctrl_cost_weight=0.1, ....)
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight,
        reset_noise_scale etc. rgb rendering comes from tracking camera (so
        agent does not run away from screen)
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks. Added
        reward_threshold to environments.
  * v0: Initial versions release (1.0.0)
  """


  def __init__(self,
               ctrl_cost_weight=0.5,
               use_contact_forces=False,
               contact_cost_weight=5e-4,
               healthy_reward=1.0,
               terminate_when_unhealthy=True,
               healthy_z_range=(0.2, 1.0),
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True,
               legacy_spring=True,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
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
    obs = self._get_obs(qp, self.sys.info(qp))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_survive': zero,
        'reward_ctrl': zero,
        'reward_contact': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)

    velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
    forward_reward = velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = (self._contact_cost_weight *
                    jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
    obs = self._get_obs(qp, info)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=qp.pos[0, 0],
        y_position=qp.pos[0, 1],
        distance_from_origin=jp.norm(qp.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe ant body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    # qpos: position and orientation of the torso and the joint angles.
    if self._exclude_current_positions_from_observation:
      qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]
    else:
      qpos = [qp.pos[0], qp.rot[0], joint_angle]

    # qvel: velocity of the torso and the joint angle velocities.
    qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    if self._use_contact_forces:
      cfrc = [
          jp.clip(info.contact.vel, -1, 1),
          jp.clip(info.contact.ang, -1, 1)
      ]
      # flatten bottom dimension
      cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
    else:
      cfrc = []

    return jp.concatenate(qpos + qvel + cfrc)

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
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
  }
}
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
joints {
  name: "$floor.pelvis"
  stiffness: 5000.0
  parent: "floor"
  child: "pelvis"
  parent_offset {
    z: 1.0
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "abdomen_x"
  stiffness: 600.0
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
  stiffness: 600.0
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
  stiffness: 600.0
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
  stiffness: 50.0
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.4477880001068115
  }
  child_offset {
    z: 0.22389400005340576
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 50.0
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.4477880001068115
  }
  child_offset {
    z: 0.22389400005340576
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 50.0
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.4477880001068115
  }
  child_offset {
    z: 0.22389400005340576
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  stiffness: 200.0
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.04809999838471413
    y: -0.36621999740600586
    z: 0.4869999885559082
  }
  child_offset {
    x: -0.024049999192357063
    y: -0.18310999870300293
    z: 0.2434999942779541
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 200.0
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.04809999838471413
    y: -0.36621999740600586
    z: 0.4869999885559082
  }
  child_offset {
    x: -0.024049999192357063
    y: -0.18310999870300293
    z: 0.2434999942779541
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 200.0
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.04809999838471413
    y: -0.36621999740600586
    z: 0.4869999885559082
  }
  child_offset {
    x: -0.024049999192357063
    y: -0.18310999870300293
    z: 0.2434999942779541
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  stiffness: 150.0
  parent: "right_upper_arm"
  child: "right_lower_arm"
  parent_offset {
    z: -0.5495759844779968
  }
  child_offset {
    z: -0.2747879922389984
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 5000.0
  parent: "right_lower_arm"
  child: "right_hand"
  parent_offset {
    z: -0.2589470148086548
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "left_shoulder_x"
  stiffness: 200.0
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.04809999838471413
    y: 0.36621999740600586
    z: 0.4869999885559082
  }
  child_offset {
    x: -0.024049999192357063
    y: 0.18310999870300293
    z: 0.2434999942779541
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 200.0
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.04809999838471413
    y: 0.36621999740600586
    z: 0.4869999885559082
  }
  child_offset {
    x: -0.024049999192357063
    y: 0.18310999870300293
    z: 0.2434999942779541
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 200.0
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.04809999838471413
    y: 0.36621999740600586
    z: 0.4869999885559082
  }
  child_offset {
    x: -0.024049999192357063
    y: 0.18310999870300293
    z: 0.2434999942779541
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  stiffness: 150.0
  parent: "left_upper_arm"
  child: "left_lower_arm"
  parent_offset {
    z: -0.5495759844779968
  }
  child_offset {
    z: -0.2747879922389984
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 5000.0
  parent: "left_lower_arm"
  child: "left_hand"
  parent_offset {
    z: -0.2589470148086548
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "right_hip_x"
  stiffness: 300.0
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.16977399587631226
  }
  child_offset {
    y: -0.08488699793815613
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 300.0
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.16977399587631226
  }
  child_offset {
    y: -0.08488699793815613
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 300.0
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.16977399587631226
  }
  child_offset {
    y: -0.08488699793815613
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  stiffness: 300.0
  parent: "right_thigh"
  child: "right_shin"
  parent_offset {
    z: -0.8430920243263245
  }
  child_offset {
    z: -0.42154601216316223
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 200.0
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.8197399973869324
  }
  child_offset {
    z: -0.4098699986934662
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 200.0
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.8197399973869324
  }
  child_offset {
    z: -0.4098699986934662
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 200.0
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.8197399973869324
  }
  child_offset {
    z: -0.4098699986934662
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  stiffness: 300.0
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.16977399587631226
  }
  child_offset {
    y: 0.08488699793815613
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 300.0
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.16977399587631226
  }
  child_offset {
    y: 0.08488699793815613
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 300.0
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.16977399587631226
  }
  child_offset {
    y: 0.08488699793815613
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  stiffness: 300.0
  parent: "left_thigh"
  child: "left_shin"
  parent_offset {
    z: -0.8430920243263245
  }
  child_offset {
    z: -0.42154601216316223
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 200.0
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.8197399973869324
  }
  child_offset {
    z: -0.4098699986934662
  }
  rotation {
    x: -0.0
    z: -0.0
  }
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
  stiffness: 200.0
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.8197399973869324
  }
  child_offset {
    z: -0.4098699986934662
  }
  rotation {
    x: -0.0
    z: 90.0
  }
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
  stiffness: 200.0
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.8197399973869324
  }
  child_offset {
    z: -0.4098699986934662
  }
  rotation {
    x: -0.0
    y: -90.0
  }
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
  angle {
  }
}
actuators {
  name: "abdomen_y"
  joint: "abdomen_y"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "abdomen_z"
  joint: "abdomen_z"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "neck_x"
  joint: "neck_x"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "neck_y"
  joint: "neck_y"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "neck_z"
  joint: "neck_z"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "right_shoulder_x"
  joint: "right_shoulder_x"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "right_shoulder_y"
  joint: "right_shoulder_y"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "right_shoulder_z"
  joint: "right_shoulder_z"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "right_elbow"
  joint: "right_elbow"
  strength: 60.0
  angle {
  }
}
actuators {
  name: "left_shoulder_x"
  joint: "left_shoulder_x"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "left_shoulder_y"
  joint: "left_shoulder_y"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "left_shoulder_z"
  joint: "left_shoulder_z"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "left_elbow"
  joint: "left_elbow"
  strength: 60.0
  angle {
  }
}
actuators {
  name: "right_hip_x"
  joint: "right_hip_x"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "right_hip_z"
  joint: "right_hip_z"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "right_hip_y"
  joint: "right_hip_y"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "right_knee"
  joint: "right_knee"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "right_ankle_x"
  joint: "right_ankle_x"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "right_ankle_y"
  joint: "right_ankle_y"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "right_ankle_z"
  joint: "right_ankle_z"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "left_hip_x"
  joint: "left_hip_x"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "left_hip_z"
  joint: "left_hip_z"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "left_hip_y"
  joint: "left_hip_y"
  strength: 125.0
  angle {
  }
}
actuators {
  name: "left_knee"
  joint: "left_knee"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "left_ankle_x"
  joint: "left_ankle_x"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "left_ankle_y"
  joint: "left_ankle_y"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "left_ankle_z"
  joint: "left_ankle_z"
  strength: 50.0
  angle {
  }
}

friction: 1.0
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "pelvis"
  second: "floor"
}
collide_include {
  first: "torso"
  second: "floor"
}
collide_include {
  first: "head"
  second: "floor"
}
collide_include {
  first: "right_upper_arm"
  second: "floor"
}
collide_include {
  first: "right_lower_arm"
  second: "floor"
}
collide_include {
  first: "right_elbow"
  second: "floor"
}
collide_include {
  first: "right_hand"
  second: "floor"
}
collide_include {
  first: "left_upper_arm"
  second: "floor"
}
collide_include {
  first: "left_lower_arm"
  second: "floor"
}
collide_include {
  first: "left_elbow"
  second: "floor"
}
collide_include {
  first: "left_hand"
  second: "floor"
}
collide_include {
  first: "right_thigh"
  second: "floor"
}
collide_include {
  first: "right_shin"
  second: "floor"
}
collide_include {
  first: "right_knee"
  second: "floor"
}
collide_include {
  first: "right_foot"
  second: "floor"
}
collide_include {
  first: "left_thigh"
  second: "floor"
}
collide_include {
  first: "left_shin"
  second: "floor"
}
collide_include {
  first: "left_knee"
  second: "floor"
}
collide_include {
  first: "left_foot"
  second: "floor"
}
dt: 0.05
substeps: 10
dynamics_mode: "legacy_spring"
"""
