from autodp import rdp_acct
from autodp import rdp_bank


def main():
    tau = 10
    poisson_mechanism = False
    # Simply use the same standard deviation of the Gaussian noise.
    # sigma = 50
    sigma = 70
    delta = 1e-6

    # Declare the moment accountant.Autodp supports a RDP (Renyi Differential
    # Privacy) based analytical Moment Accountant, which allows us to track the RDP
    # for each query conveniently.
    acct = rdp_acct.anaRDPacct()

    gaussian = lambda x: rdp_bank.RDP_inde_pate_gaussian(
        params={'sigma': int(sigma / tau)}, alpha=x)

    if poisson_mechanism:
        private_query_count = 110
        sampling_probability = 1.0
        acct.compose_poisson_subsampled_mechanisms(
            gaussian, prob=sampling_probability, coeff=private_query_count)
    else:
        private_query_count = 215
        acct.compose_mechanism(func=gaussian, coeff=private_query_count)

    # compute privacy loss
    epsilon = acct.get_eps(delta)
    print(f"Composition of the mechanisms gives: ({epsilon}, {delta})-DP")


if __name__ == "__main__":
    main()
