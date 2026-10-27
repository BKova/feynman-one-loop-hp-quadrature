# Numerical evaluation of one-loop integrals using quadrature rules

This repository contains the research implementation accompanying the preprint
[*Numerical evaluation of one-loop integrals using quadrature rules*](PREPRINT_URL).
It includes the numerical examples discussed in the paper, with reference
results available in [results.txt](results.txt).

## Requirements

- **Python 3.13.5 or later**.
- Python packages listed in [requirements.txt](requirements.txt).
- **Git** to clone the repository and initialize the optional submodule.

The additional requirements depend on the examples being run:

| Examples | Additional requirements |
| --- | --- |
| `regular`, `threshold`, and `higher coefficients` | None beyond the Python dependencies. |
| `numerator` | The `mpreal` submodule, a GNU C++ compiler (`g++`) with C++17 support, and the MPFR and GMP development libraries. |

The commands below assume a Linux environment with a Bash-compatible shell.
System package installation commands are provided for Ubuntu.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BKova/feynman-one-loop-hp-quadrature.git one-loop-quadrature
cd one-loop-quadrature
```

Run all subsequent commands from the repository root.

### 2. Set up the Python environment

Create and activate a virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Activate the environment again with `source .venv/bin/activate` whenever you
open a new shell to run the examples.

This completes the setup for the `regular`, `threshold`, and
`higher coefficients` examples.

### 3. Build the numerator library (optional)

Complete this step to run the `numerator` examples.

Initialize the `mpreal` submodule:

```bash
git submodule update --init --recursive
```

Install the compiler and development libraries on Ubuntu:

```bash
sudo apt update
sudo apt install g++ libmpfr-dev libgmp-dev
```

On other Linux distributions, install the equivalent packages using your
distribution's package manager.

Compile the high-precision numerator implementation:

```bash
g++ -O3 -std=c++17 -fPIC -shared \
    ./box_numerator/box_numerator.cpp \
    -o ./box_numerator.so \
    -Wl,-z,defs -lmpfr -lgmp
```

This creates `box_numerator.so` in the repository root.

## Running the examples

With the virtual environment activated, invoke the entry point as follows:

```text
python main.py <cpu_count> [<box_numerator_lib>]
```

| Argument | Required | Description |
| --- | --- | --- |
| `cpu_count` | Yes | Positive integer specifying the number of worker subprocesses used for the computations. |
| `box_numerator_lib` | For `numerator` examples | Path to the compiled numerator shared library, such as `./box_numerator.so`. |

Run with four worker subprocesses, without the optional numerator library:

```bash
python main.py 4
```

Run with four worker subprocesses and the numerator library:

```bash
python main.py 4 ./box_numerator.so
```

Use a worker count appropriate for the CPU resources available to you.
More than one worker is recommended because some examples require substantial
computation.

## Reference results

The results for the examples presented in the paper are provided in
[results.txt](results.txt).