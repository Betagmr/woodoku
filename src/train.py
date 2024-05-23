from src.environment.woodoku_env import WoodokuEnv


if __name__ == "__main__":
    env = WoodokuEnv(render_mode="ansi")
    observation, info = env.reset()
    for i in range(100000):
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        env.render()

        if terminated:
            env.reset()

    env.close()
