import numpy as np
from scipy.special import gammaln, comb
import math
import torch


def stable_logsumexp(x):
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))


def stable_logsumexp_two(x, y):
    a = np.maximum(x, y)
    if np.isneginf(a):
        return a
    else:
        return a + np.log(np.exp(x - a) + np.exp(y - a))


def stable_log_diff_exp(x, y):
    # ensure that y > x
    # this function returns the stable version of log(exp(y)-exp(x)) if y > x
    if y > x:
        s = True
        mag = y + np.log(1 - np.exp(x - y))
    elif y < x:
        s = False
        mag = x + np.log(1 - np.exp(y - x))
    else:
        s = True
        mag = -np.inf

    return s, mag


def stable_sum_signed(xs, x, ys, y):
    # x and y are log abs,  xs,ys are signs
    if xs == ys:
        s = ys
        out = stable_logsumexp_two(x, y)
    elif ys:
        s, out = stable_log_diff_exp(x, y)
    else:
        s, out = stable_log_diff_exp(y, x)
    return s, out


def stable_inplace_diff_in_log(vec, signs, n=-1):
    """ This function replaces the first n-1 dimension of vec with the log of abs difference operator
     Input:
        - `vec` is a numpy array of floats with size larger than 'n'
        - `signs` is a numpy array of bools with the same size as vec
        - `n` is an optional argument in case one needs to compute partial differences
            `vec` and `signs` jointly  describe a vector of real numbers' sign and abs in log scale.
     Output:
        The first n-1 dimension of vec and signs will store the log-abs and sign of the difference.

     """
    #
    # And the first n-1 dimension of signs with the sign of the differences.
    # the sign is assigned to True to break symmetry if the diff is 0
    # Input:
    assert (vec.shape == signs.shape)
    if n < 0:
        n = np.max(vec.shape) - 1
    else:
        assert (np.max(vec.shape) >= n + 1)
    for j in range(0, n, 1):
        if signs[j] == signs[j + 1]:  # When the signs are the same
            # if the signs are both positive, then we can just use the standard one
            signs[j], vec[j] = stable_log_diff_exp(vec[j], vec[j + 1])
            # otherwise, we do that but toggle the sign
            if signs[j + 1] == False:
                signs[j] = ~signs[j]
        else:  # When the signs are different.
            vec[j] = stable_logsumexp_two(vec[j], vec[j + 1])
            signs[j] = signs[j + 1]


def get_forward_diffs(fun, n):
    """ This is the key function for computing up to nth order forward difference evaluated at 0"""
    # Pre-compute the finite difference operators
    # Save them in log-scale
    func_vec = np.zeros(n + 3)  # _like(self.CGFs_int, float)
    signs_func_vec = np.ones(n + 3, dtype=bool)
    deltas = np.zeros(
        n + 2)  # ith coordinate of deltas stores log(abs(ith order discrete derivative))
    signs_deltas = np.zeros(n + 2, dtype=bool)
    for i in range(1, n + 3, 1):
        func_vec[i] = fun(1.0 * (i - 1))
    for i in range(0, n + 2, 1):
        # Diff in log scale
        # tmp = np.diff((signs_func_vec-0.5)*2 * np.exp(func_vec))

        stable_inplace_diff_in_log(func_vec, signs_func_vec, n=n + 2 - i)

        # tmp2 = (signs_func_vec-0.5)*2 *  np.exp(func_vec)

        deltas[i] = func_vec[0]
        signs_deltas[i] = signs_func_vec[0]
    return deltas, signs_deltas


def get_forward_diffs_naive(fun, n):
    func_vec = np.zeros(n + 3)  # _like(self.CGFs_int, float)
    signs_func_vec = np.ones(n + 3, dtype=bool)
    deltas = np.zeros(
        n + 2)  # ith coordinate of deltas stores log(abs(ith order discrete derivative))
    signs_deltas = np.zeros(n + 2, dtype=bool)
    for i in range(0, n + 3, 1):
        func_vec[i] = np.exp(fun(1.0 * (i - 1)))
    for i in range(0, n + 2, 1):
        # Diff in log scale
        # tmp = np.diff((signs_func_vec-0.5)*2 * np.exp(func_vec))

        func_vec = np.diff(func_vec)
        # tmp2 = (signs_func_vec-0.5)*2 *  np.exp(func_vec)

        deltas[i] = np.log(np.abs(func_vec[0]))
        signs_deltas[i] = (func_vec[0] >= 0)
    return deltas, signs_deltas


def get_forward_diffs_direct(fun, n):
    func_vec = np.zeros(n + 3)  # _like(self.CGFs_int, float)
    signs_func_vec = np.ones(n + 3, dtype=bool)
    deltas = np.zeros(
        n + 2)  # ith coordinate of deltas stores log(abs(ith order discrete derivative))
    signs_deltas = np.zeros(n + 2, dtype=bool)
    signs_deltas_out = np.zeros(n + 2, dtype=bool)
    for i in range(0, n + 3, 1):
        func_vec[i] = fun(1.0 * (i - 1))

    C_stirling = np.zeros(n + 3)
    anchor_point = np.zeros(n + 3)
    for i in range(0, n + 2, 1):
        # i+1 choose 0 to i+1 choose (i+1)/2
        coeff = fun(1.0 * (i)) / (i + 1)
        func1 = lambda x: x * coeff + np.log(1 - np.exp(fun(x - 1) - x * coeff))

        for j in range(1, i + 1, 1):
            C_stirling[j] = np.log(comb(i + 1, j))  # logcomb(i+1,j)
        for j in range(1, i + 1, 1):
            anchor_point[j] = func1(j)
        tmp = anchor_point[0:(i + 1) + 1] + C_stirling[0:(i + 1) + 1]

        # Examples of these coefficients
        # -1, 1                   i+1= 1  odd
        # 1，-2， 1               i+1 = 2  even
        # -1，3，-3，1            i+1 = 3 odd
        # 1，-4，6，-4，1          i+1 = 4 even
        signs_deltas[i] = True
        deltas[i] = -np.inf
        for j in range(0, i + 1, 2):
            s, out = stable_log_diff_exp(tmp[j], tmp[j + 1])
            signs_deltas[i], deltas[i] = stable_sum_signed(signs_deltas[i],
                                                           deltas[i], s, out)

        if not (i + 1) % 2:  # even
            signs_deltas[i] = ~signs_deltas[i]
            signs_deltas[i], deltas[i] = stable_sum_signed(signs_deltas[i],
                                                           deltas[i], False,
                                                           tmp[i + 1])

        #
        # Lastly toggle the sign and add the anchor point back

        signs_deltas_out[i], deltas[i] = stable_sum_signed(~signs_deltas[i],
                                                           deltas[i],
                                                           True, np.log(
                np.exp(coeff) - 1) * (i + 1))
    return deltas, signs_deltas_out


def logcomb(n, k):
    return (gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1))


def get_binom_coeffs(sz):
    C = np.zeros(shape=(sz + 1, sz + 1));
    # for k in range(1,sz + 1,1):
    #    C[0, k] = -np.inf
    for n in range(sz + 1):
        C[n, 0] = 0  # 1
    for n in range(1, sz + 1, 1):
        C[n, n] = 0
    for n in range(1, sz + 1, 1):
        for k in range(1, n, 1):
            # numerical stable way of implementing the recursion rule
            C[n, k] = stable_logsumexp_two(C[n - 1, k - 1], C[n - 1, k])
    # only the lower triangular part of the matrix matters
    return C


def get_binom_coeffs_dict(sz):
    C = {}  # np.zeros(shape = (sz + 1, sz + 1));
    # for k in range(1,sz + 1,1):
    #    C[0, k] = -np.inf
    for n in range(sz + 1):
        C[(n, 0)] = 0  # 1
    for n in range(1, sz + 1, 1):
        C[(n, n)] = 0
    for n in range(1, sz + 1, 1):
        for k in range(1, n, 1):
            # numerical stable way of implementing the recursion rule
            C[(n, k)] = stable_logsumexp_two(C[(n - 1, k - 1)], C[(n - 1, k)])
    # only the lower triangular part of the matrix matters
    return C


def expand_binom_coeffs_dict(C, sz, sznew):
    for n in range(sz, sznew + 1, 1):
        C[(n, 0)] = 0
    for n in range(sz, sznew + 1, 1):
        C[(n, n)] = 0
    for n in range(sz, sznew + 1, 1):
        for k in range(1, n, 1):
            C[(n, k)] = stable_logsumexp_two(C[(n - 1, k - 1)], C[(n - 1, k)])
    return C  # also need to update the size of C to sznew whenever this function is called just to keep track.


def RDP_linear_interpolation(func, x):
    # linear interpolation upper bound through the convexity of CGF
    epsinf = func(np.inf)

    if np.isinf(x):
        return epsinf
    if (x >= 1.0) and (x <= 2.0):
        return np.minimum(epsinf, func(2.0))
    if np.equal(np.mod(x, 1), 0):
        return np.minimum(epsinf, func(x))
    xc = math.ceil(x)
    xf = math.floor(x)
    return np.minimum(
        epsinf,
        ((x - xf) * (xc - 1) * func(xc) + (1 - (x - xf)) * (xf - 1) * func(
            xf)) / (x - 1)
    )


def tau_limit(labels, tau):
    """
    Clip votes to max tau positive labels per record.

    For multi-label problem, limit the attribute of each neighbor to be
    smaller than tau, where tau could be served as a composition coefficient.

    :param labels: predicted labels by each teacher of size number of teachers x
    number of labels (for the multi-label classification).
    :param tau: the value of tau for the tau-approximation where we limit the
    sensitivity of a given teacher by limiting his positive votes to tau.

    :return: clipped votes
    """
    votes = np.zeros_like(labels)
    for idx in range(len(labels)):
        record = labels[idx]
        if np.sum(record) == 0:
            votes[idx] = record
        else:
            votes[idx] = record * min(tau / float(np.sum(record)), 1)
    # votes = np.sum(votes, axis=0)
    return votes


def clip_votes(votes, tau):
    """
    Clip votes to max tau positive labels per record.

    For multi-label problem, limit the attribute of each neighbor to be
    smaller than tau, where tau could be served as a composition coefficient.

    :param votes: predicted labels by each teacher of size number of teachers x
    number of labels (for the multi-label classification).
    :param tau: the value of tau for the tau-approximation where we limit the
    sensitivity of a given teacher by limiting his positive votes to tau.

    :return: clipped votes
    """
    votes = votes.astype(np.float)
    sums = np.sum(votes, axis=-1)
    multiply = tau / np.maximum(tau, sums)
    multiply = np.expand_dims(multiply, axis=1)
    votes *= multiply
    return votes


def clip_votes_tensor(votes, tau, norm):
    """
    Clip votes to max tau positive labels per record.

    For multi-label problem, limit the attribute value of each neighbor to be
    smaller than tau, where tau could be served as a composition coefficient.

    :param votes: predicted labels by each teacher of size number of teachers x
        number of labels (for the multi-label classification).
    :param tau: the value of tau for the tau-approximation where we limit the
        sensitivity of a given teacher by limiting his positive votes to tau.
    :param norm: L norm for clipping.

    :return: clipped votes
    """
    if norm == '1':
        votes = clip_votes_tensor_l1(
            votes=votes, tau=tau)
    elif norm == '2':
        votes = clip_votes_tensor_l2(
            votes=votes, tau=tau)
    else:
        raise Exception(f"Unsupported norm for clipping votes: {norm}.")
    return votes


def clip_votes_tensor_l1(votes, tau):
    """
    Clip votes to max tau positive labels per record.

    For multi-label problem, limit the attribute value of each neighbor to be
    smaller than tau, where tau could be served as a composition coefficient.

    :param votes: predicted labels by each teacher of size number of teachers x
        number of labels (for the multi-label classification).
    :param tau: the value of tau for the tau-approximation where we limit the
        sensitivity of a given teacher by limiting his positive votes to tau.

    :return: clipped votes
    """
    # print('votes: ', votes)
    votes = votes.to(torch.float32)
    sums = torch.sum(votes, dim=-1)
    ones = torch.ones_like(sums)
    multiply = tau / torch.maximum(tau * ones, sums)
    multiply = torch.unsqueeze(multiply, -1)
    votes *= multiply
    # print('tau votes: ', votes)
    return votes


def clip_votes_tensor_l2(votes, tau):
    """
    Clip votes to max tau L2 norm.

    For multi-label problem, limit the attribute of each neighbor to be
    smaller than tau, where tau could be served as a composition coefficient.

    :param votes: predicted labels by each teacher of size number of teachers x
        number of labels (for the multi-label classification).
    :param tau: the value of tau for the tau-approximation where we limit the
        sensitivity of a given teacher by limiting his positive votes to tau.

    :return: clipped votes
    """
    # print('votes: ', votes)
    votes = votes.to(torch.float32)
    squares = torch.square(votes)
    sums = torch.sum(squares, dim=-1)
    norms = torch.sqrt(sums)
    votes_max = torch.maximum(torch.tensor(1), norms / tau)
    votes_max = torch.unsqueeze(votes_max, -1)
    votes = votes / votes_max
    return votes


if __name__ == "__main__":
    votes = np.ones([10, 40])
    # votes = np.zeros([10, 40])
    # votes = np.random.random([10, 40]) * 10
    # votes = tau_limit(labels=votes, tau=10)
    votes = clip_votes(votes=votes, tau=5)
    print('votes: ', votes)

    norm = '1'

    votes = torch.tensor([[1, 2, 3], [5, 0, 0]])
    votes = clip_votes_tensor(votes=votes, tau=5, norm=norm)
    print('votes: ', votes)

    votes = torch.tensor([[6, 2, 3], [5, 0, 0], [0, 6, 0]])
    votes = clip_votes_tensor(votes=votes, tau=5, norm=norm)
    print('votes: ', votes)
    print('votes sum rows: ', votes.sum(dim=-1))

    norm = '2'
    print('norm: ', norm)

    votes = torch.tensor([[1, 1, 1], [1, 0, 0]])
    votes = clip_votes_tensor(votes=votes, tau=1, norm=norm)
    print('votes: ', votes)

    votes = torch.tensor(
        [[6, 2, 3], [5, 0, 0], [0, 6, 0], [1, 2, 1], [3, 3, 3]])
    votes = clip_votes_tensor(votes=votes, tau=5, norm=norm)
    print('votes: ', votes)
    print('votes sum rows: ', votes.sum(dim=-1))
