import numpy as np
from copy import copy

# TODO other mutation types
# TODO other missions with current
# TODO different autopilots


def random_mutation(vec, probability, distribution, scalar, random_generator):
    """ replace random cells """
    new_vec = copy(vec)
    num_vals = len(vec)
    num_to_mutate = round(num_vals * probability)

    is_mutated = random_generator.random((num_vals, 1)) < probability
    num_to_mutate = np.sum(is_mutated)

    if num_to_mutate > 0:
        new_vals = get_random_vals(
            distribution, 1, num_to_mutate, rng=random_generator, scalar=scalar
        )

        new_vec[np.squeeze(is_mutated)] = np.squeeze(new_vals)

    return new_vec


def get_random_vals(method, num_row, num_col, rng, scalar=1):
    """ return random sample from different population methods """
    if method == "rand":
        return rng.random((num_row, num_col)) * scalar
    elif method == "randn":
        return rng.standard_normal((num_row, num_col)) * scalar
    elif method == "zeros":
        return np.zeros((num_row, num_col))
    elif method == "randpm":
        return (2 * rng.random((num_row, num_col)) - 1) * scalar
    else:
        # this exception should be raised in the weights/bias functions beforehand
        raise ValueError("Unknown method ('rand','randn','randpm','zeros')")


if __name__ == "__main__":
    x_vector = np.array(np.arange(0, 10.0, 1))
    seed = np.random.choice(10000, 1)
    y_vector = random_mutation(x_vector, 0.5, "randn", 5.0, seed)
    for x, y in zip(x_vector, y_vector):
        print(f"{x:.2f} , {y:.2f}")
