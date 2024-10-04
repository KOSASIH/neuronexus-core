import numpy as np
from shor import Shor

def main():
    shor = Shor(5)
    outcome = shor.shor()
    print(outcome)

if __name__ == "__main__":
    main()
