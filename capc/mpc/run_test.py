import os
import numpy as np
import threading

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class start_party(threading.Thread):
    def __init__(self, thread_name, cmd):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.cmd = cmd

    def run(self):
        print(self.thread_name)
        os.system(self.cmd)


def int_laplacian(size, loc=0.0, scale=1.0):
    noise = np.random.laplace(loc=loc, scale=scale, size=size)
    noise = noise * (1e6)
    return noise


def main():
    try:
        client_sum = [2, 1, 2, 1]
        csp_sum = [1, 1, 2, 3]
        csp_sum = csp_sum  # + int_laplacian(size=len(csp_sum))

        client_thread = start_party(
            "Thread-1",
            "./build/bin/sum_histogram 1 12345 {} {}".format(
                str(len(client_sum)), ",".join(map(str, client_sum))
            ),
        )
        client_thread.start()

        csp_thread = start_party(
            "Thread-2",
            "./build/bin/sum_histogram 2 12345 {} {}".format(
                str(len(csp_sum)), ",".join(map(str, csp_sum))
            ),
        )
        csp_thread.start()

        client_thread.join()
        csp_thread.join()

        histogram = []
        with open("histogram.txt") as file:
            for line in file:
                msg = int(line.strip())
                histogram.append(msg)

        print(histogram)
    except Exception as ex:
        print("not able to start thread: ", ex)
        logger.exception(ex)


if __name__ == "__main__":
    main()
