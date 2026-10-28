def test_pid():
    from gym_pybullet_drones.examples.pid import run
    run(gui=False, plot=False, output_folder='tmp')

def test_pid_velocity():
    from gym_pybullet_drones.examples.pid_velocity import run
    run(gui=False, plot=False, output_folder='tmp')

def test_downwash():
    from gym_pybullet_drones.examples.downwash import run
    run(gui=False, plot=False, output_folder='tmp')

def test_learn():
    from gym_pybullet_drones.examples.learn import run
    run(gui=False, plot=False, output_folder='tmp', local=False)

def test_pid_record():
    from gym_pybullet_drones.examples.pid import run
    run(gui=False, plot=False, record_video=True, duration_sec=1, output_folder='tmp')
