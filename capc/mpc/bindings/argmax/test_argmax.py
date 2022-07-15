import cython_argmax
import argparse


def main(party, port, values):
    result = cython_argmax.pyargmax(party=party, port=port, values=values)
    print("result: ", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cython Test Argmax")
    parser.add_argument(
        "--party",
        type=int,
        default=1,
        help="The number of the party: "
        "1 - Alice (answering party), "
        "2 - Bob (querying party)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="The port number for the network communication between the " "parties.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=int,
        default=[3, 4, 1],
        help="The logits or noise from one of the parties.",
    )
    main(party=party, port=port, values=values)
