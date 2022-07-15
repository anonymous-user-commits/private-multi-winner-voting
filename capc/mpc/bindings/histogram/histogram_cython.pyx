"""Cython interface definition for histogram via MPC."""
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "histogram_cython.hpp":
    # Each party provides below information.
    #
    # 1st compute sum of votes across the parties - the histogram.
    # Histogram has to be aggregated across the parties - done in MPC.
    # Results the vector of length: num_classes.
    #
    # 2nd compute the max of histogram and compare with:
    # threshold T - sum(noise_threshold from each party)
    #
    # How much noise each party should provide in noise_threshold as well as in
    # the noise_argmax? sigma for each party = \sqrt(n) * global_sigma, where n
    # is the number of parties.
    # sigma_global**2 = n * sigma_local**2
    # sigma_local = sigma_global / \sqrt(n)
    # Use the aggregated noise for the threshold T as well as the noise_argmax.
    #
    # TODO:
    # (Sample from normal distribution. Can a local noise reveal any information
    # about the global distribution?)
    # There might be a technical question here - resolve after the whole
    # protocol is designed.
    long long argmax(
            int party,
            int num_classes,  # In our example = 10 (classes, e.g. MNIST)
            int port,
            vector[string] ip_address,
            float threshold,  # parameter T
            # noise for the max number of votes from a single teacher
            float noise_threshold,
            vector[float] noise_argmax,  # noise for the vector of votes (histogram)
            vector[int] votes,  # one-hot vector (the vote from the party/teacher)
    )

def pyargmax(party: int,
             port: int,
             ip_address: string,
             sigma_threshold: float,
             threshold: float,
             sigma_gnmax: float,
             values: list):
    return argmax(party=party,
                  port=port,
                  ip_address=ip_address,
                  sigma_threshold=sigma_threshold,
                  threshold=threshold,
                  sigma_gnmax=sigma_gnmax,
                  values=values)
