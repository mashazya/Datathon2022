"""
Script to measure the quality of the solutions provided.
Uses the input file to extract the driver pins and the pins to route.
It will then perform the following:
 - Check all the pins are routed.
 - Check all the chains are valid (starts with 1 driver input and ends with 1 driver output)
 - Count the number of chains made.
 - Measure the length of every chain.

Call the script using
    `python solution_metrics.py input_file_name output_file_name`
"""

import collections
import sys
import traceback

Point = collections.namedtuple("Point", ["x", "y"])

Pin = collections.namedtuple("Pin", ["name", "loc"])


# ---------------------------------------------------------------------------------------------------------------------


def extract_pins(input_file):
    """Extracts the list of pins (drivers and pins to route) from the input file.

    Args:
        input_file {str} -- The name of the input file.

    Returns:
        (list(Pin), list(Pin)) -- The list of driver pins and the list of pins to route.
    """
    header_done = False
    # previous_was_row = False
    previous_was_die_area = False

    driver_pins = []
    pins_to_route = []

    driver_name = ""
    ongoing_driver = 0  # Used to skip lines

    with open(input_file, "r") as f:
        for line in f.readlines():
            # Ignore the header and the rows:
            if not header_done:
                if line == "\n" and previous_was_die_area:
                    header_done = True
                if line == "\n":
                    continue
                if line.split()[0] == "DIEAREA":
                    previous_was_die_area = True

            if line == "\n":
                continue

            # Extract driver pins:
            if "DRIVERPIN_" in line:
                driver_name = line.split()[1]
                ongoing_driver = 2
                continue
            if ongoing_driver > 1:
                ongoing_driver -= 1
                continue
            if ongoing_driver == 1:
                p = Point(*[int(s) for s in line.split() if s.isdigit()])
                driver_pins.append(Pin(driver_name, p))
                ongoing_driver -= 1

            # Extract pins:
            if "im_psyched" in line:
                sp = line.split()
                p = Point(*[int(s) for s in sp if s.isdigit()])
                pins_to_route.append(Pin(sp[0], p))

    return driver_pins, pins_to_route


# ---------------------------------------------------------------------------------------------------------------------


def extract_links(output_file):
    """Extracts the links from the generated output file. They might not be in order
    so this is just extracting the links.

    Args:
        output_file {str}: The name of the output file to parse.

    Returns:
        dict(str: str) -- The links between the pins.
    """
    links = {}
    conn_in = ""

    with open(output_file, "r") as f:
        for line in f.readlines():
            # Skip the semi-colon lines if they are separated:
            if line.strip() == ";":
                continue

            # Skip the net names, not used:
            if line[0] == "-":
                continue

            # Get conn_in
            if conn_in == "":
                conn_in = line.split()[1]
            # Get conn_out
            else:
                conn_out = line.split()[1]
                links[conn_in] = conn_out
                conn_in = ""

    return links


# ---------------------------------------------------------------------------------------------------------------------


def extract_chains(links, pins_index):
    """Extract the chains of pins from the output file.

    Args:
        links {dict(str: str)} -- The links between the pins extracted from the ouput.
        pins_index {dict(str: Pin)} -- The index of pins to route and driver pins based on their name.

    Raises:
        ValueError -- If the number of driver pins used is not right or if a drive pin is used multiple times.

    Returns:
        list(list(Pin)) -- The list of chains of pins.
    """
    chains = []

    # Search for the driver inputs:
    inputs = [p for p in links.keys() if pin_is_input_driver(p)]
    outputs = [p for p in links.keys() if pin_is_output_driver(p)]

    if not len(inputs) != len(outputs):
        raise ValueError("The same number of input and ouput pins should be used from the driver.")

    if not (2 <= len(inputs) <= 16):
        raise ValueError(f"The number of chains should be between 2 and 16, currently {len(input)}.")
    print("Valid number of driver pins used: check")

    if len(inputs) != len(set(inputs)) or len(outputs) != len(set(outputs)):
        raise ValueError("Driver pins should be used only once each maximum.")
    print("Driver pins are used only once: check")

    # Follow the trail until it reaches a driver output:
    for i in inputs:
        chain = []
        chain.append(pins_index[i])
        next_pin = pins_index[i]

        while not pin_is_output_driver(next_pin):
            next_pin = pins_index[links[next_pin.name]]
            chain.append(next_pin)

        chains.append(chain)

    return chains


# ---------------------------------------------------------------------------------------------------------------------


def check_valid_chain(chain):
    """Checks that the provided chain is valid (i.e. starts with an input pin, ends with
    an output pin).

    Args:
        chain {list(str)} -- The chain to test.

    Returns:
        bool -- Returns True if the chain is valid, False otherwise.
    """
    return pin_is_input_driver(chain[0]) and pin_is_output_driver(chain[-1])


# ---------------------------------------------------------------------------------------------------------------------


def pin_is_input_driver(pin):
    """Check if the provided pin is an driver input pin.

    Args:
        pin {Pin | str}: The pin to check.

    Returns:
        bool -- Returns True if the pin is a driver input pin, False otherwise.
    """
    if isinstance(pin, Pin):
        pin = pin.name

    if "DRIVERPIN_" not in pin:
        return False
    return int(pin.split("_")[1]) <= 15


# ---------------------------------------------------------------------------------------------------------------------


def pin_is_output_driver(pin):
    """Check if the provided pin is an driver output pin.

    Args:
        pin {Pin | str}: The pin to check.

    Returns:
        bool -- Returns True if the pin is a driver output pin, False otherwise.
    """
    if isinstance(pin, Pin):
        pin = pin.name

    if "DRIVERPIN_" not in pin:
        return False
    return int(pin.split("_")[1]) > 15


# ---------------------------------------------------------------------------------------------------------------------


def check_all_pins_routed(chains, pins_to_route):
    """Checks that all the pins to route are part of a chain.
    And that each pin is routed only once.

    Args:
        chains {list(list(Pin))} -- The list of chains of pins.
        pins_to_route {list(Pin)} -- The list of pins to route.

    Returns:
        bool -- Returns True if all pins are routed, False otherwise.
    """
    to_route = set([p.name for p in pins_to_route])
    found = set()

    for chain in chains:
        for pin in chain:
            if pin_is_input_driver(pin) or pin_is_output_driver(pin):
                continue

            if pin.name in found:
                return False
            found.add(pin.name)

    return to_route == found


# ---------------------------------------------------------------------------------------------------------------------


def manhattan_distance(a, b):
    """Computes the Manhattan distance between two points."""
    return abs(a.x - b.x) + abs(a.y - b.y)


# ---------------------------------------------------------------------------------------------------------------------


def measure_chain_length(chain):
    """Measures the length of the provided chain based on the location of the pins.

    Args:
        chain {list(Pin)} -- A chain of pins.

    Return:
        int -- The length of the provided chain based on Manhattan's distance.
    """
    distance = 0
    for i in range(len(chain) - 1):
        distance += manhattan_distance(chain[i].loc, chain[i + 1].loc)
    return distance


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # Parse input and output files:
    driver_pins, pins_to_route = extract_pins(sys.argv[1])

    try:
        links = extract_links(sys.argv[2])
        print("Output file formatted properly: check")
    except Exception as e:
        print(
            "Encountered an exception while trying to parse the output:\n",
            traceback.print_exception(e),
            "\nThis solution does not meet the required format.",
        )
        exit(1)

    # Index the data:
    pin_locations = {}
    for pin in driver_pins:
        pin_locations[pin.name] = pin.loc
    for pin in pins_to_route:
        pin_locations[pin.name] = pin.loc

    pins_index = {}
    for pin in driver_pins:
        pins_index[pin.name] = pin
    for pin in pins_to_route:
        pins_index[pin.name] = pin

    # Extract the chains and checks they're okay:
    try:
        chains = extract_chains(links, pins_index)
        print("Chains could be extracted: check")

        for chain in chains:
            assert check_valid_chain(chain), "Chains should start and end at the driver."
        print("Chains start and end at the driver: check")

        assert check_all_pins_routed(chains, pins_to_route), "All pins should be routed exactly once."
        print("All pins are routed exactly once: check")
    except Exception as e:
        print(
            "Encountered an exception while trying to extract the chains:\n",
            traceback.print_exception(e),
            "\nThis solution does not meet the required constraints on the chains.",
        )
        exit(1)

    lengths = []
    print("-" * 40)
    print(f"Number of chains formed: {len(chains)}")
    for i, chain in enumerate(chains):
        lengths.append(measure_chain_length(chain))
        print(f" - Chain {i} - Length = {lengths[-1]}")
    print(f"Average length = {sum(lengths)/len(lengths)}")
