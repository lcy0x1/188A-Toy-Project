from gym.envs.registration import register

register(
    id='vehicle-v0',
    entry_point='gym_vehicle.envs:VehicleEnv',
)