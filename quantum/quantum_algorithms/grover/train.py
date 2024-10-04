import numpy as np
from grover import Grover

def main():
    grover = Grover(5)
    outcome = grover.grover()
    print(outcome)

if __name__ == "__main__":
    main()
