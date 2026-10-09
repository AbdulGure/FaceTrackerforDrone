import multiprocessing as mp
import time

from simulation.drone_simulator import simulator_process


def main():
    sim_queue = mp.Queue(maxsize=2)

    process = mp.Process(
        target=simulator_process,
        args=(sim_queue,)
    )

    process.start()

    try:
        for i in range(100):
            sim_queue.put({
                "yaw": 0.4,
                "vertical": 0.2,
                "drone_state": "FOLLOWING",
                "identity_state": "VERIFIED"
            })

            time.sleep(0.05)

    finally:
        sim_queue.put(None)
        process.join(timeout=3)

        if process.is_alive():
            process.terminate()
            process.join()

        sim_queue.close()

    print("Simulator exit code:", process.exitcode)


if __name__ == "__main__":
    mp.freeze_support()
    main()